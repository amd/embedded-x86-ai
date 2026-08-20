<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# cuda-to-hip-porting: End-to-End Walkthrough

This walkthrough demonstrates the **cuda-to-hip-porting** skill from start to finish on a
real CUDA sample project. Follow these steps in order.

---

## Prerequisites

| Requirement | Value |
|-------------|-------|
| Skill | `cuda-to-hip-porting` (see [skills/cuda-to-hip-porting/SKILL.md](../skills/cuda-to-hip-porting/SKILL.md)) |
| ROCm version | 7.2+ (required for gfx1151 / Strix Halo) |
| GPU arch | Run `amdgpu-arch` to detect yours (e.g. `gfx1151`, `gfx1100`, `gfx90a`) |
| Docker image | `rocm/dev-ubuntu-24.04:7.2-complete` (gfx1151) or `rocm/rocm-terminal:latest` (other) |
| hipify-perl | Included in ROCm; verify with `hipify-perl --version` |

---

## Step 0: Profile the CUDA Project (Optional but Recommended)

Run the profiler to understand scope before converting anything:

```bash
# From the repo root — path works whether the skill lives in amd/skills or amd_skills/
python3 skills/cuda-to-hip-porting/scripts/profile_cuda_project.py /path/to/cuda/project \
  --json thisproject.json \
  --md   thisproject.md
```

The profiler identifies blockers (inline PTX, WMMA, CDP, graphics interop), required libraries,
and estimates porting effort. Output guides your conversion strategy.

**When to use:** Unknown codebases, projects with 50+ `.cu` files, or library-heavy code.

---

## Step 1: Setup Checklist (Do This FIRST)

Skipping these steps causes cascading build failures.

### 1a. Check available Docker images

```bash
# Avoid a 15 GB pull if an image already exists
docker images | grep rocm
```

### 1b. Create `hip_helper.h`

This file **must exist** before running hipify:

```bash
# Copy the pre-built helper from the skill reference
# Ask the skill to create hip_helper.h from reference/hip_helper.md:
# "Read reference/hip_helper.md and write the C++ code block to Common/hip_helper.h"

# Also update existing Common/ headers
sed -i 's|cuda_runtime.h|hip/hip_runtime.h|g' Common/helper_math.h
sed -i 's|cuda_runtime.h|hip/hip_runtime.h|g' Common/helper_cusolver.h  # if it exists
```

### 1c. Always use `--user` in Docker

```bash
# CORRECT — files remain owned by you
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete bash
```

---

## Step 2: hipify-perl Conversion

### Single file

```bash
hipify-perl input.cu > output.hip
```

### Whole project (parallel — 10-50× faster for 50+ files)

```bash
CORES=$(nproc)
find . -name "*.cu" -print0 | \
    xargs -0 -P $CORES -I {} bash -c 'hipify-perl "{}" > "{}.hip" 2> "{}.warnings.txt"'

# Review conversion warnings in aggregate
cat **/*.warnings.txt | grep -v "^$" | sort | uniq -c | sort -rn | head -30
```

---

## Step 3: Apply Proactive Fixes (S67–S71)

Run this **immediately after hipify, before the first build**. It fixes the most common
post-hipify issues automatically and raises first-attempt build success to 70–85%.

```bash
# Full script in reference/proactive-fixes.md — copy it to your project root
./proactive_fixes.sh /path/to/converted/project
```

What the script does:
- **S67** — Replaces hardcoded warp size `32` → `warpSize` (prevents 27× numerical errors on AMD wave64)
- **S68** — Wraps `cooperative_groups.h` includes with `#ifdef __HIP_PLATFORM_AMD__`
- **S69** — Adds `typedef hipTextureObject_t cudaTextureObject_t;` where needed
- **S70** — Comments out `hipProfilerStart/Stop()` (deprecated in ROCm 7.2+)
- **S71** — Detects kernel-only files and multi-file project directories

---

## Step 4: Build with hipcc

### Auto-detect GPU and compile

```bash
GPU=$(amdgpu-arch 2>/dev/null | head -1)
echo "Target GPU: ${GPU:-none detected}"

if [ -n "$GPU" ]; then
    hipcc -I./Common --offload-arch=$GPU -w -o sample sample.hip
else
    # No GPU present — compile-only to check for errors
    hipcc -I./Common -c -w -o sample.o sample.hip
    echo "Compile-only mode (no GPU detected)"
fi
```

### CMake projects (better success rate)

```bash
cmake -B build -DCMAKE_CXX_COMPILER=hipcc
cmake --build build --parallel $(nproc)
```

### Expected success rates

| Scenario | Success Rate |
|----------|-------------|
| hipify completes | 95–100% of files |
| Standalone build (files with `main()`) | 5–10% |
| CMake multi-file build (portable files) | 40–50% |
| With proactive fixes (S67–S71) applied | 70–85% |

Kernel-only files (no `main()`) failing standalone is **normal and expected** — they need
linking with a companion main file.

---

## Step 5: Verify on AMD GPU

```bash
# Run the built binary
./sample

# For Docker environments
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete ./sample
```

### Proof-of-HIP validation (S62)

Verify the `.hip` files were actually compiled (not auto-hipified from `.cu` at build time):

```bash
# Move .cu files out of tree, then build — proves .hip files are in use
mv /path/to/cuda/project/src src_cuda_backup
cmake --build build --parallel $(nproc)
# Build succeeds → .hip files confirmed
```

---

## Troubleshooting

| Error | Action |
|-------|--------|
| `'checkCudaErrors' undeclared` | `hip_helper.h` missing — see Step 1b |
| `cooperative_groups.h not found` | Apply S68 fix — see Step 3 |
| `hipProfilerStart` returns error | Apply S70 fix — see Step 3 |
| `undefined reference to 'main'` | Normal for kernel-only files — link with main |
| Build succeeds but wrong results | Check S67 (wave size 32 → warpSize) |
| gfx1151 crashes at runtime | Requires ROCm 7.2+ — see Prerequisites |

For full error patterns and fixes, see:
- [reference/troubleshooting.md](../skills/cuda-to-hip-porting/reference/troubleshooting.md)
- [reference/docker-setup.md](../skills/cuda-to-hip-porting/reference/docker-setup.md)
- [reference/api-rewrites.md](../skills/cuda-to-hip-porting/reference/api-rewrites.md)
