<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# cuda-to-hip-porting: Example AI Prompts

Copy-paste these into a `claude` session running in the project directory.
The `cuda-to-hip-porting` skill must be installed (see top-level README).

---

## A. First port - "do the whole thing"

Use this on any CUDA project. Best first run: a simple vectorAdd-style project.

> Use the cuda-to-hip-porting skill to port this project from CUDA to HIP.
> Steps I want you to follow:
>   0. [OPTIONAL] Run skills/cuda-to-hip-porting/scripts/profile_cuda_project.py to understand what CUDA
>      features and libraries are present. Save output to thisproject.md/json.
>   1. Complete the CRITICAL SETUP CHECKLIST before any conversion:
>      - Check available Docker images (avoid 15GB download delays)
>      - Create Common/hip_helper.h from skill reference
>      - Update Common/helper_math.h: cuda_runtime.h → hip/hip_runtime.h
>      - Always use --user $(id -u):$(id -g) in Docker commands
>   2. Run hipify on .cu files and save output as .hip files.
>      - For 50+ files: use parallel processing (xargs -P or GNU parallel)
>      - Check available cores with nproc, use 80-90% for maximum throughput
>      - Save warnings to per-file *.warnings.txt files
>   3. Read every hipify warning. For any warning that hipify did not
>      auto-rewrite, apply the manual rewrite recipe from the skill's
>      reference files and explain what you changed and why.
>   4. Build the original CUDA version with nvcc and the new HIP version
>      with hipcc. Fix build errors as they appear.
>      - Expected standalone build success: 5-10% (only files with main)
>      - Kernel-only files failing to link is NORMAL and EXPECTED
>      - For better success rate (40-50%), use CMake for multi-file linking
>   5. Run both binaries on small input and diff the output. Report any
>      numerical differences.
>   6. Summarize with separate metrics:
>      - Conversion success: X/Y files converted (expect 95-100%)
>      - Complete programs built: Count only files with main() (expect 5-10%)
>      - Kernel-only files: Count separately (expected to fail standalone)
>      - Which APIs translated cleanly vs required manual rewrites
>      - Any unresolved gaps or non-portable features

---

## B. Just profile this codebase - no port yet

Use this on an unfamiliar CUDA codebase before deciding to port. Recommended for
unknown codebases, large projects (50+ files), and library-heavy code.

> Run skills/cuda-to-hip-porting/scripts/profile_cuda_project.py on this directory and save output:
>   python3 skills/cuda-to-hip-porting/scripts/profile_cuda_project.py . --json thisproject.json --md thisproject.md
>
> Then read the output and tell me:
>   - Which NVIDIA libraries are in use, and what their ROCm equivalents are.
>   - Which CUDA features (inline PTX, WMMA, warp-sync, cooperative groups,
>     Unified Memory, texture objects, Driver API, NVML) are present.
>   - Which features are blockers (cannot convert) vs require manual work.
>   - Based on file count, recommend whether to use parallel processing (50+ files).
>   - Estimate difficulty: trivial / moderate / hard / research-grade. Justify
>     the rating with expected conversion success rate and build success rate.
>   - Note: Conversion success (95-100%) ≠ build success (5-10% standalone, 40-50% CMake)

---

## C. Port only what hipify cannot handle

Useful when hipify has already been run and you want focused help on the
warnings.

> Read hipify_stats.txt (or stderr.txt) in this directory. For each warning
> that hipify did not auto-rewrite, locate the relevant code in the .cu
> source, apply the manual rewrite recipe from the cuda-to-hip-porting skill,
> and produce the corresponding .hip code. Do not touch anything hipify
> already translated cleanly.

---

## D. Verify a hand-written port against the CUDA baseline

Useful when you (or a previous engineer) already produced a .hip file and
want a second opinion.

> Compare the CUDA source (.cu) and the HIP port (.hip) in this directory.
> Identify:
>   - Any CUDA construct in the .cu that has no corresponding rewrite in the
>     .hip (silent omission).
>   - Any rewrite in the .hip that does not preserve the semantics of the
>     original (wrong translation).
>   - Any place where the port hard-codes a 32-thread warp assumption that
>     will fail on CDNA wave64.
>   - Any place where the port silently changes numerical behavior
>     (atomic scope, fence type, rounding mode, mixed precision).
> For each finding, cite the line numbers in both files.

---

## E. Port with Auto GPU Detection

Use this for any AMD GPU - auto-detects architecture.

> Port this project to HIP with auto GPU detection.
>   - Use ROCm 7.2+ for gfx1151 (rocm/dev-ubuntu-24.04:7.2-complete)
>   - Check warpSize assumptions (don't hardcode 32)
>   - Create hip_helper.h if needed (see reference/api-rewrites.md)
> Build: `GPU=$(amdgpu-arch | head -1) && hipcc -I. --offload-arch=$GPU -w -o output input.hip`
> If no GPU: `hipcc -I. -c -w -o output.o input.hip` (compile-only)

---

## F. Demonstrate a specific gap (teaching mode)

Useful for onboarding a new colleague.

> I want to see why the cuda-to-hip-porting skill calls out S25 (Tensor Core /
> WMMA) as a Critical gap. Run hipify on a WMMA sample (e.g., from NVIDIA cuda-samples).
> Show me the warnings hipify emits. Then walk through the rocWMMA rewrite
> step by step, pointing out each place where the AMD fragment layout
> differs from the NVIDIA WMMA fragment layout.

---

## G. Apply to a real customer project (sanitized)

Use this when picking up a customer codebase. Replace `<PATH>`.

> Treat `<PATH>` as a customer CUDA codebase I want to evaluate for
> porting to ROCm. Do not modify any files in that tree.
>   1. Run the profiler and save output (thisproject.md/json).
>   2. Open the top three most CUDA-feature-dense files and summarize what
>      each does and which cuda-to-hip-porting recipes will apply.
>   3. Produce a one-page migration scoping note with REALISTIC metrics:
>      - Conversion success estimate: X/Y files (expect 95-100%)
>      - Portable files: Count excluding graphics/PTX/platform-specific
>      - Expected build success: 5-10% standalone, 40-50% with CMake
>      - Kernel-only files: Count separately (will fail standalone - NORMAL)
>      - Blockers: PTX, WMMA, CDP, graphics interop (cannot convert)
>      - Manual effort: Libraries, cooperative groups, textures (hours/weeks)
>      - What hipify will handle automatically
>      - Recommended approach: sequential vs parallel processing
>      - Difficulty rating with justification
> Output the scoping note as Markdown.

---

## H. Large Project Port (100+ files)

Use this for large CUDA projects with many files. Combines profiling,
parallelization, setup checklist, and realistic expectations.

> Use the cuda-to-hip-porting skill to port this large CUDA project to HIP.
> This project has 100+ files, so use PARALLELIZATION for maximum efficiency.
>
> Steps I want you to follow:
>   1. Profile the project to understand scope:
>      - Run skills/cuda-to-hip-porting/scripts/profile_cuda_project.py and save thisproject.md/json
>      - Report total .cu files, libraries used, and blockers identified
>
>   2. Complete CRITICAL SETUP CHECKLIST:
>      - Check available Docker images: docker images | grep rocm
>      - Prefer existing images to avoid 15GB download
>      - Create Common/hip_helper.h from skill reference
>      - Update Common/helper_math.h and other Common/*.h files
>      - Always use --user $(id -u):$(id -g) in Docker commands
>
>   3. Parallel hipify conversion:
>      - Check available cores: nproc
>      - Use 80-90% of cores for maximum throughput
>      - Convert ALL .cu files in parallel (not a sample):
>        find . -name "*.cu" -print0 | xargs -0 -P $(nproc) -I {} bash -c '
>          hipify-perl "{}" > "{}.hip" 2> "{}.warnings.txt"
>        '
>      - Collect all warnings: cat **/*.warnings.txt > all_warnings.txt
>
>   4. Aggregate warning analysis:
>      - Categorize warnings by type (WMMA, PTX, unsupported APIs, etc.)
>      - Count occurrences of each category
>      - Apply manual rewrites by category (not per-file)
>      - Summarize what was fixed and why
>
>   5. Build with REALISTIC EXPECTATIONS:
>      - Understand file types: complete programs vs kernel-only files
>      - Expected standalone success: 5-10% (only files with main)
>      - Kernel-only files failing is NORMAL and EXPECTED
>      - For CMake projects: expect 40-50% success rate
>      - Use parallel building where possible
>
>   6. Report statistics SEPARATELY:
>      - Conversion success: X/Y files converted (expect 95-100%)
>      - Portable files: Count excluding graphics/PTX/platform-specific
>      - Complete programs built: Count only files with main() (5-10%)
>      - Kernel-only files: Count separately (expected to fail standalone)
>      - Non-portable features: Graphics interop, PTX, etc. (15-20%)
>      - Manual fixes applied: Summarized by category
>      - Processing time saved: Sequential vs parallel comparison

---

## Tips

- The skill assumes you are running from inside the project directory so
  it can resolve relative paths to `.cu` files. `cd` first.
- If hipify is not on PATH, the skill will try the Docker image in
  `skills/cuda-to-hip-porting/reference/docker-setup.md`. Bring that up first.
- For library-heavy ports (anything pulling in cuBLASLt, cuDNN, NCCL),
  pass `USE_CUDNN=1` or equivalent flags explicitly - the skill will not
  guess your build configuration.
- **For gfx1151:** ROCm 7.2+ required (earlier versions crash at runtime).
- **Key skills to know:** S3 (API signatures), S8 (cooperative groups), S11 (libraries),
  S18 (64-bit warp masks), S25 (WMMA→rocWMMA), S48 (OpenCV CUDA), S49 (helper macros),
  S51 (NVTX→ROCTX). See `reference/api-rewrites.md`.
- If the model gets confused, the single best recovery prompt is:
  "Stop. Re-read the cuda-to-hip-porting skill's reference/api-rewrites.md
  and tell me which recipe applies here before changing any code."
