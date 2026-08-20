#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
profile_cuda_project.py

Static analyzer that fills out the AMD "CUDA-to-ROCm Migration - Customer
Discovery Questionnaire" for a single CUDA project on disk. Designed to be
run repeatedly across the seven investigation samples and, later, against
real customer codebases.

Usage:
    python3 profile_cuda_project.py <project_dir> [--json out.json] [--md out.md]

Strategy: walk the source tree, classify file extensions, grep for NVIDIA
library headers, link flags, and CUDA feature patterns (inline PTX, WMMA,
warp-sync primitives, Graphs, Unified Memory, texture objects, NCCL,
NVTX, NVRTC, Driver API, etc.). Report against the 10 questionnaire
sections.

Inputs are file system + source text only - no execution, no network.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp",
               ".py", ".f", ".f90", ".rs"}
# Files we will grep for patterns:
CODE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp"}
# Files counted toward CUDA-specific LOC:
CUDA_EXTS = {".cu", ".cuh"}
SKIP_DIRS = {".git", ".hg", "build", "dist", "out", "__pycache__",
             "node_modules", ".venv", "venv", "hip_output"}
# Filename substrings indicating a port artifact, not original CUDA source.
PORT_NAME_MARKERS = ("_hip", ".hip.", "_rocm")

# Library detection: header include -> library name, importance bucket.
LIB_PATTERNS = [
    ("cublasLt", r"\bcublasLt(?:_v2)?\.h\b|\bcublasLt[A-Z]"),
    ("cuBLAS",   r"\bcublas(?:_v2)?\.h\b|\bcublas[A-Z]"),
    ("cuDNN",    r"\bcudnn\.h\b|\bcudnn[A-Z_]"),
    ("cuFFT",    r"\bcufft(?:Xt)?\.h\b|\bcufft[A-Z]"),
    ("cuSPARSE", r"\bcusparse\.h\b|\bcusparse[A-Z]"),
    ("cuSPARSELt", r"\bcusparseLt\.h\b|\bcusparseLt[A-Z]"),
    ("cuSOLVER", r"\bcusolver[A-Za-z]*\.h\b|\bcusolver[A-Z]"),
    ("cuRAND",   r"\bcurand(?:_kernel)?\.h\b|\bcurand[A-Z]"),
    ("NCCL",     r"\bnccl\.h\b|\bncclAll|\bncclReduce|\bncclGroup|\bncclComm"),
    ("Thrust",   r"\bthrust/"),
    ("CUB",      r"\bcub/cub\.h\b|\bcub/"),
    ("CUTLASS",  r"\bcutlass/"),
    ("TensorRT", r"\bNvInfer\.h\b|\bnvinfer1::"),
    ("cuQuantum", r"\bcustatevec\.h\b|\bcutensornet\.h\b"),
    ("nvJPEG",   r"\bnvjpeg\.h\b|\bnvjpeg[A-Z]"),
    ("nvCOMP",   r"\bnvcomp\.h\b|\bnvcomp::"),
    ("Optical Flow SDK", r"\bnvOpticalFlow|\bNvOFAPI"),
    ("DALI",     r"\bdali/"),
    ("NVTX",     r"\bnvToolsExt(?:Cuda)?\.h\b|\bnvtx[A-Z]|\bNVTX_"),
    ("NVML",     r"\bnvml\.h\b|\bnvml[A-Z]"),
    ("MPI",      r"\bmpi\.h\b|\bMPI_[A-Z]"),
]

# CUDA feature detection.
FEATURE_PATTERNS = [
    ("Inline PTX",                r"\basm\s*(?:volatile\s*)?\("),
    ("WMMA C++ API",              r"nvcuda::wmma|wmma::fragment|wmma::load_matrix_sync|wmma::mma_sync"),
    ("mma.sync PTX",              r"mma\.sync|mma\.m\d+n\d+k\d+"),
    ("Warp-sync primitives",      r"__shfl_(?:sync|xor_sync|up_sync|down_sync)|__ballot_sync|__activemask|__any_sync|__all_sync|__match_(?:any|all)_sync"),
    ("Cooperative groups",        r"cooperative_groups|cg::|cooperative_groups::"),
    ("CUDA Graphs",               r"cudaGraph_t|cudaStreamBeginCapture|cudaGraphLaunch|cudaGraphInstantiate"),
    ("Dynamic Parallelism",       r"<<<[^>]+>>>\s*[^;]*;.*<<<", ),  # weak; also flag by attribute
    ("Unified Memory",            r"cudaMallocManaged|cudaMemAdvise|cudaMemPrefetchAsync"),
    ("Texture / surface objects", r"cudaTextureObject_t|cudaCreateTextureObject|texture<|surface<|cudaResourceDesc"),
    ("NVRTC",                     r"\bnvrtc(?:Program|Compile|GetPTX|Result)\b"),
    ("Driver API",                r"\bcuCtx[A-Z]|\bcuModule[A-Z]|\bcuMemMap|\bcuMemSetAccess|\bcuLaunchKernel\b"),
    ("Cached load/store intrinsics", r"__ldg\b|__ldcs\b|__ldcv\b|__stcs\b|__stwt\b"),
    ("Atomic with explicit scope", r"atomic_ref|cuda::atomic|cuda::memory_order|__threadfence_system|__threadfence_block"),
    ("Mixed precision (fp16/bf16)", r"__half\b|__nv_bfloat16|__hfma|__hadd|cublasComputeType_t|CUBLAS_COMPUTE_"),
    ("Shared memory",             r"__shared__\b"),
]

LANG_BY_EXT = {
    ".c":    "C",
    ".cc":   "C++",
    ".cpp":  "C++",
    ".cxx":  "C++",
    ".h":    "C/C++ headers",
    ".hpp":  "C++ headers",
    ".cu":   "CUDA C/C++",
    ".cuh":  "CUDA C/C++ headers",
    ".py":   "Python",
    ".f":    "Fortran",
    ".f90":  "Fortran",
    ".rs":   "Rust",
}

BUILD_FILES = {
    "Makefile":        "Make",
    "makefile":        "Make",
    "GNUmakefile":     "Make",
    "CMakeLists.txt":  "CMake",
    "BUILD":           "Bazel",
    "BUILD.bazel":     "Bazel",
    "WORKSPACE":       "Bazel",
    "meson.build":     "Meson",
    "setup.py":        "Python setuptools",
    "pyproject.toml":  "Python (PEP 517)",
}

CI_FILES = {
    ".github/workflows": "GitHub Actions",
    ".gitlab-ci.yml":    "GitLab CI",
    "Jenkinsfile":       "Jenkins",
    ".circleci":         "CircleCI",
    "azure-pipelines.yml": "Azure Pipelines",
    ".buildkite":        "Buildkite",
}


def is_port_artifact(p: Path) -> bool:
    n = p.name.lower()
    return any(m in n for m in PORT_NAME_MARKERS)


def iter_source_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SOURCE_EXTS and not is_port_artifact(p):
                yield p


def count_lines(p: Path) -> int:
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT  = re.compile(r"//[^\n]*")


def strip_c_comments(text: str) -> str:
    """Remove /* ... */ and // ... comments so pattern matches reflect code,
    not header comments listing what the file demonstrates."""
    text = _BLOCK_COMMENT.sub(" ", text)
    text = _LINE_COMMENT.sub(" ", text)
    return text


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def read_code(p: Path) -> str:
    """Like read_text, but strips comments for code files."""
    return strip_c_comments(read_text(p))


def detect_build(root: Path):
    found = []
    for name, kind in BUILD_FILES.items():
        if (root / name).exists():
            found.append((name, kind))
    return found


def detect_ci(root: Path):
    found = []
    for rel, name in CI_FILES.items():
        if (root / rel).exists():
            found.append(name)
    return found


def detect_tests(root: Path) -> list[str]:
    matches = []
    for p in iter_source_files(root):
        n = p.name.lower()
        if n.startswith("test_") or n.endswith("_test.cu") or n.endswith("_test.cpp"):
            matches.append(str(p.relative_to(root)))
    return matches


def detect_hipify_artifacts(root: Path) -> dict:
    out = {}
    for name in ("hipify_stats.txt", "hipify_stderr.txt"):
        p = root / name
        if p.exists():
            out[name] = p.read_text(encoding="utf-8", errors="ignore").strip()
    # presence of *.hip files alongside *.cu suggests a port has been done
    cu  = [p for p in root.rglob("*.cu") if not is_port_artifact(p)]
    hip = list(root.rglob("*.hip"))
    out["cu_files"]  = len(cu)
    out["hip_port_files_present"] = len(hip)
    return out


def detect_target_arches(root: Path) -> list[str]:
    """Look for -arch / -gencode / --offload-arch flags in build files and scripts.
    Honors SKIP_DIRS so we don't pick up identifiers from preprocessed / port-artifact
    files (e.g. `hipblasStrsm_64` matching sm_64)."""
    arch_pat = re.compile(r"\b(sm_\d{2,3}|compute_\d{2,3}|gfx\w+)\b")
    # Only look at flag contexts so we don't grab API-name false positives.
    flag_pat = re.compile(
        r"(?:-arch[= ]|-gencode[= ]|--offload-arch[= ]|--cuda-gpu-arch[= ]|"
        r"CUDA_ARCHITECTURES|AMDGPU_TARGETS|HIP_ARCHITECTURES)[^\n]*"
    )
    archs = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in {".mk", ".cmake", ".sh"} or p.name in BUILD_FILES:
                text = read_text(p)
                for line in flag_pat.findall(text):
                    archs.update(arch_pat.findall(line))
    return sorted(archs)


def profile(project_dir: Path) -> dict:
    root = project_dir.resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    files = list(iter_source_files(root))
    by_ext: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_ext[p.suffix.lower()].append(p)

    total_loc = 0
    cuda_loc  = 0
    per_lang_loc: Counter = Counter()
    for ext, paths in by_ext.items():
        lang = LANG_BY_EXT.get(ext, ext)
        for p in paths:
            n = count_lines(p)
            total_loc += n
            per_lang_loc[lang] += n
            if ext in CUDA_EXTS:
                cuda_loc += n

    # Read all code-file text with comments stripped so pattern matches
    # reflect actual code, not banner / docstring mentions.
    blobs = []
    for p in files:
        if p.suffix.lower() in CODE_EXTS:
            blobs.append((p, read_code(p)))

    # Library hits.
    lib_hits = {}
    for name, pattern in LIB_PATTERNS:
        rgx = re.compile(pattern)
        matches = []
        for p, text in blobs:
            if rgx.search(text):
                matches.append(str(p.relative_to(root)))
        if matches:
            lib_hits[name] = matches

    # Feature hits.
    feat_hits = {}
    for name, pattern in FEATURE_PATTERNS:
        rgx = re.compile(pattern)
        matches = []
        for p, text in blobs:
            if rgx.search(text):
                matches.append(str(p.relative_to(root)))
        if matches:
            feat_hits[name] = matches

    builds = detect_build(root)
    cis    = detect_ci(root)
    tests  = detect_tests(root)
    hipify = detect_hipify_artifacts(root)
    arches = detect_target_arches(root)

    cuda_pct = round(100.0 * cuda_loc / total_loc, 1) if total_loc else 0.0

    size_bucket = (
        "< 50,000 lines" if total_loc < 50_000 else
        "50,000 - 500,000 lines" if total_loc < 500_000 else
        "500,000 - 2,000,000 lines" if total_loc < 2_000_000 else
        "> 2,000,000 lines"
    )

    return {
        "project": root.name,
        "path": str(root),
        "totals": {
            "files": len(files),
            "total_loc": total_loc,
            "cuda_loc": cuda_loc,
            "cuda_pct": cuda_pct,
            "size_bucket": size_bucket,
            "per_language_loc": dict(per_lang_loc.most_common()),
            "files_by_ext": {ext: len(ps) for ext, ps in sorted(by_ext.items())},
        },
        "build_systems": [k for _, k in builds],
        "build_files":   [n for n, _ in builds],
        "ci":            cis,
        "tests":         tests,
        "target_arches": arches,
        "libraries":     lib_hits,
        "features":      feat_hits,
        "hipify_artifacts": hipify,
    }


# -------- Reporting --------

def render_markdown(r: dict) -> str:
    out = []
    out.append(f"# Questionnaire Profile - {r['project']}")
    out.append(f"_Path: `{r['path']}`_\n")

    t = r["totals"]
    out.append("## Section 1 - Codebase basics")
    out.append(f"- Files (source): **{t['files']}**, total LOC: **{t['total_loc']:,}**")
    out.append(f"- CUDA-specific LOC: **{t['cuda_loc']:,}** ({t['cuda_pct']}%)")
    out.append(f"- Size bucket: **{t['size_bucket']}**")
    out.append(f"- Per-language LOC: " +
               ", ".join(f"{k}: {v:,}" for k, v in t["per_language_loc"].items()))
    out.append(f"- File-extension counts: " +
               ", ".join(f"{k or '<none>'}: {v}" for k, v in t["files_by_ext"].items()))
    out.append("")

    out.append("## Section 2 - Languages / frameworks")
    langs = list(t["per_language_loc"].keys())
    out.append(f"- Languages detected: **{', '.join(langs) if langs else 'none'}**")
    # Heuristic framework signal from libraries section:
    fw = []
    if "TensorRT" in r["libraries"]: fw.append("TensorRT")
    if any(k.startswith("py") for k in t["per_language_loc"]): fw.append("Python (possibly PyTorch/TF)")
    out.append(f"- Framework signals: {', '.join(fw) if fw else 'none detected statically'}")
    out.append("")

    out.append("## Section 3 - NVIDIA libraries in use")
    if r["libraries"]:
        for name, paths in r["libraries"].items():
            shown = ", ".join(paths[:3]) + (" ..." if len(paths) > 3 else "")
            out.append(f"- **{name}** - in: {shown}")
    else:
        out.append("- None detected.")
    out.append("")

    out.append("## Section 4 - CUDA features used")
    if r["features"]:
        for name, paths in r["features"].items():
            shown = ", ".join(paths[:3]) + (" ..." if len(paths) > 3 else "")
            out.append(f"- **{name}** - in: {shown}")
    else:
        out.append("- None detected (runtime API only, no special features).")
    out.append("")

    out.append("## Section 5 - Hardware target")
    out.append(f"- Target architectures referenced in build: " +
               (", ".join(r["target_arches"]) if r["target_arches"] else "none found"))
    out.append("- Deployment shape: **unknown from source alone** (ask customer).")
    out.append("")

    out.append("## Section 6 - Build / test / ops")
    out.append(f"- Build system(s): {', '.join(r['build_systems']) or 'none detected'}")
    out.append(f"- Build files: {', '.join(r['build_files']) or '-'}")
    out.append(f"- CI: {', '.join(r['ci']) or 'none detected'}")
    out.append(f"- Test files: {len(r['tests'])}" +
               (f" (e.g. {', '.join(r['tests'][:3])})" if r["tests"] else ""))
    out.append("")

    out.append("## Section 7 - Perf / correctness")
    out.append("- Static-only profile; perf bar and golden data must come from the customer.")
    if "NVTX" in r["libraries"]:
        out.append("- NVTX annotations present - app is already instrumented for profilers.")
    out.append("")

    out.append("## Section 8 / 9 / 10 - Constraints, engagement, free-text")
    out.append("- Cannot be inferred from source. Ask customer.")
    out.append("")

    out.append("## Hipify port artifacts (if present)")
    h = r["hipify_artifacts"]
    out.append(f"- `.cu` files (CUDA source): {h.get('cu_files', 0)}; "
               f"`.hip` files present from prior port: {h.get('hip_port_files_present', 0)}")
    for k in ("hipify_stats.txt", "hipify_stderr.txt"):
        if k in h:
            out.append(f"- **{k}** present (first 240 chars):")
            snippet = h[k][:240].replace("\n", " ").strip()
            out.append(f"  > {snippet}")
    out.append("")

    # Tech-plan gap mapping.
    out.append("## Tech-plan gap mapping (Section 3 of the plan)")
    mapping = {
        "3.1 Inline PTX":            "Inline PTX" in r["features"],
        "3.2 Tensor Core / MMA":     ("WMMA C++ API" in r["features"]) or ("mma.sync PTX" in r["features"]) or ("cublasLt" in r["libraries"]) or ("cuDNN" in r["libraries"]),
        "3.3 Warp / wave semantics": "Warp-sync primitives" in r["features"],
        "3.4 Memory model / atomics": "Atomic with explicit scope" in r["features"],
        "3.5 CUDA Graphs":           "CUDA Graphs" in r["features"],
        "3.6 Unified Memory":        "Unified Memory" in r["features"],
        "3.7 Libraries (any)":       any(k in r["libraries"] for k in
                                         ("cuBLAS", "cublasLt", "cuDNN", "cuFFT",
                                          "cuSPARSE", "cuSPARSELt", "cuSOLVER",
                                          "cuRAND", "NCCL", "CUTLASS", "TensorRT",
                                          "cuQuantum", "nvJPEG", "nvCOMP",
                                          "Optical Flow SDK", "DALI", "NVTX", "NVML")),
        "3.8 Profiling (NVTX)":      "NVTX" in r["libraries"],
        "3.10 Driver API / NVML":    ("Driver API" in r["features"]) or ("NVML" in r["libraries"]),
    }
    for k, v in mapping.items():
        out.append(f"- {k}: {'**hit**' if v else 'not detected'}")
    out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_dir")
    ap.add_argument("--json", help="Write JSON to this path")
    ap.add_argument("--md",   help="Write Markdown report to this path")
    args = ap.parse_args()

    result = profile(Path(args.project_dir))
    md = render_markdown(result)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    if args.md:
        Path(args.md).write_text(md)

    # Always print the markdown to stdout for human review.
    print(md)


if __name__ == "__main__":
    main()
