---
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
name: cuda-to-hip-porting
version: "1.0.0"
description: >-
  Ports NVIDIA CUDA codebases to AMD HIP/ROCm end-to-end: runs hipify-perl,
  applies proactive wave-size and cooperative-groups fixes, maps libraries
  (cuBLAS→hipBLAS, cuFFT→hipFFT, cuDNN→MIOpen), configures Docker and ROCm
  for the target GPU arch, and migrates ML inference (ONNX→MIGraphX, TensorRT
  plugins). Use when the user's request involves: hipify, CUDA-to-HIP, ROCm
  porting, .cu to .hip, hipcc, HIP migration, CUDAExtension, MIOpen tuning,
  gfx1151, or Strix Halo.
---

<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# CUDA to HIP Porting

Port NVIDIA CUDA codebases to AMD HIP/ROCm. Auto-detects GPU architecture and applies 71 documented porting patterns end-to-end.

---

## TL;DR

**IMPORTANT: Always use Docker for CUDA-to-HIP porting.** Host ROCm installations may have version mismatches or missing tools.

```bash
# Run everything inside ROCm Docker container (use latest official ROCm)
# Check available images: docker images | grep rocm
# Pull latest: docker pull rocm/dev-ubuntu-24.04:latest
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace -w /workspace \
  rocm/dev-ubuntu-24.04:latest bash -c '
    # 1. Convert
    hipify-perl input.cu > output.hip
    
    # 2. Copy hip_helper.h (see reference/hip_helper.md)
    
    # 3. Apply S70 if needed (comment out hipProfilerStart/Stop)
    sed -i "s/hipProfilerStart()/\/\/ hipProfilerStart() \/\/ S70: deprecated/g" output.hip
    sed -i "s/hipProfilerStop()/\/\/ hipProfilerStop() \/\/ S70: deprecated/g" output.hip
    
    # 4. Build and run
    hipcc --offload-arch=$(amdgpu-arch | head -1) -I. -o output output.hip && ./output
'
```

For details on each step, see sections below.

---

## Triggers

This skill activates on:
- `.cu` or `.hip` files
- Keywords: hipify, ROCm, hipcc, CUDA-to-HIP migration
- TensorRT plugin decomposition, `cv::cuda::` types
- PyTorch `CUDAExtension`, NVIDIA Warp, CUDA graph capture
- `onnx2torch`, MIOpen tuning, Strix Halo

---

## CRITICAL SETUP CHECKLIST

**Do these FIRST before any conversion or building** (skip these → cascading failures):

### 1. Check Available Docker Images

```bash
# Check what's already downloaded (avoid 15GB delay)
docker images | grep rocm

# Recommended: Use latest official ROCm dev image
# - rocm/dev-ubuntu-24.04:latest (full dev tools, ~20GB)
# - rocm/dev-ubuntu-22.04:latest (alternative for older systems)

# Pull latest official ROCm image:
docker pull rocm/dev-ubuntu-24.04:latest
```

### 2. Create hip_helper.h (CRITICAL)

**This MUST exist before running hipify** - otherwise all builds will fail:

```bash
# Create Common/hip_helper.h from cuda-to-hip-porting skill reference
# See reference/api-rewrites.md for full content

# Also update Common/helper_math.h:
sed -i 's|cuda_runtime.h|hip/hip_runtime.h|g' Common/helper_math.h

# Also update Common/helper_cusolver.h if it exists:
sed -i 's|cuda_runtime.h|hip/hip_runtime.h|g' Common/helper_cusolver.h
```

### 3. Always Use --user Flag in Docker

```bash
# WRONG - creates root-owned files
docker run --rm -v $(pwd):/workspace rocm/... hipcc ...

# CORRECT - files owned by you
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace rocm/... hipcc ...
```

See `reference/docker-setup.md` for permission management details.

### 4. Footgun Environment Variables

Set these to avoid silent failures:

```bash
# CRITICAL for Docker: Some images don't set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Required for Strix Halo (gfx1151) on ROCm 7.2
export HSA_OVERRIDE_GFX_VERSION=11.5.1   # only if native gfx1151 support not yet in ROCm build

# MIOpen autotuning (set BEFORE importing torch — first run is slow, caches results)
export MIOPEN_FIND_MODE=3                 # 3=search, 4=search+persist, 1=cache only
export MIOPEN_FIND_ENFORCE=3

# Restrict visible GPUs (optional — avoid if unsure)
# export ROCR_VISIBLE_DEVICES=0
```

---

## Quick Start

### 1. Detect GPU Architecture

```bash
# Auto-detect GPU (returns gfx1151, gfx90a, gfx1100, etc.)
GPU=$(amdgpu-arch 2>/dev/null | head -1)
echo "Detected: ${GPU:-No GPU found}"

# Alternative methods
rocminfo | grep -m1 "gfx"
rocm-smi --showproductname
```

### 2. Convert and Build

```bash
# Convert CUDA to HIP
hipify-perl input.cu > output.hip

# Auto-detect GPU, compile, and run
GPU=$(amdgpu-arch 2>/dev/null | head -1)
if [ -n "$GPU" ]; then
    hipcc -I. --offload-arch=$GPU -w -o output output.hip && ./output
else
    echo "No GPU detected, compile-only mode"
    hipcc -I. -c -w -o output.o output.hip
fi
```

### 3. Smart Recompile-and-Run

Automatically recompiles if binary was built for different GPU:

```bash
run_hip() {
    local src="$1" bin="${1%.hip}_hip"
    local current_gpu=$(amdgpu-arch 2>/dev/null | head -1)
    
    if [ -z "$current_gpu" ]; then
        echo "No GPU detected, compile-only mode"
        hipcc -I. -c -w -o "${src%.hip}.o" "$src"
        return
    fi
    
    # Check if binary exists and matches current GPU
    local needs_rebuild=0
    if [ ! -f "$bin" ]; then
        needs_rebuild=1
    else
        local bin_arch=$(strings "$bin" 2>/dev/null | grep -oE 'gfx[0-9]+' | head -1)
        if [ "$bin_arch" != "$current_gpu" ]; then
            echo "Recompiling: binary=$bin_arch, GPU=$current_gpu"
            needs_rebuild=1
        fi
    fi
    
    [ "$needs_rebuild" -eq 1 ] && hipcc -I. --offload-arch=$current_gpu -w -o "$bin" "$src"
    ./"$bin"
}

# Usage
run_hip myapp.hip
```

### 4. Docker One-Liner

```bash
# Auto-detect GPU in Docker, recompile if needed, run
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e LD_LIBRARY_PATH=/opt/rocm/lib \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:latest bash -c '
        GPU=$(amdgpu-arch | head -1)
        if [ -z "$GPU" ]; then
            hipcc -I. -c -w -o output.o output.hip
            echo "Compile-only (no GPU)"
        else
            hipcc -I. --offload-arch=$GPU -w -o output output.hip && ./output
        fi
    '
```

---

## Conversion Workflow

```
CUDA Project (.cu files)
    ↓
0. [OPTIONAL] Profile project (skills/cuda-to-hip-porting/scripts/profile_cuda_project.py)
    ↓ Recommended for: unknown codebases, large projects (50+ files), library-heavy code
    ↓ Output: thisproject.md/json → guides conversion strategy
    ↓
1. [CRITICAL] Complete setup checklist (see above)
    ↓ - Check Docker images, create hip_helper.h, update Common/*.h
    ↓
2. hipify-perl conversion (100+ regex patterns)
    ↓ Use parallel processing for 50+ files (see Large Project Conversion)
    ↓
3. Copy hip_helper.h (replaces helper_cuda.h)
    ↓
4. Add stub main() if kernel-only file (S29)
    ↓
5. Merge multi-file projects (S16)
    ↓
6. Detect GPU: amdgpu-arch
    ↓
7. Compile: --offload-arch=$GPU (or -c if no GPU)
    ↓ Expected success rate: 5-10% standalone, 40-50% with CMake (see Build Expectations)
    ↓
8. If error → apply fix patterns → retry (up to 5x)
    ↓
9. Run on target GPU
```

**Key workflow enhancements**:
- Step 0 profiling helps scope effort before starting
- Step 1 setup checklist prevents cascading failures
- Step 2 uses parallelization for large projects (10-50x speedup)
- Step 7 sets realistic expectations (kernel-only files will fail standalone)

---

## Proactive Fix Automation (S67-S71)

**CRITICAL: Apply these fixes AFTER hipify but BEFORE building.** Raises first-attempt success to 70–85%.

| Skill | Fix | Impact |
|-------|-----|--------|
| S67 | `threadIdx.x % 32` → `warpSize` | Prevents 27× numerical errors on AMD wave64 |
| S68 | Wrap `cooperative_groups.h` with `#ifdef __HIP_PLATFORM_AMD__` | Fixes header not found |
| S69 | Add `typedef hipTextureObject_t cudaTextureObject_t` | Fixes type mismatches |
| S70 | Comment out `hipProfilerStart/Stop()` | Fixes `hipErrorNotSupported` in ROCm 7.2+ |
| S71 | Detect kernel-only files and multi-file dirs | Prevents mis-reported build failures |

```bash
# Run the complete fix script immediately after hipify
./proactive_fixes.sh /path/to/project
```

See `reference/proactive-fixes.md` for the full `proactive_fixes.sh` script and per-fix commands.

**Updated workflow:**
```
hipify-perl → proactive_fixes.sh → hipcc --offload-arch=$GPU
```

---

## Large Project Conversion (50+ files)

Parallel processing gives 10–50× speedup. See `reference/proactive-fixes.md` for full commands.

```bash
# Convert all .cu files in parallel
find . -name "*.cu" -print0 | \
    xargs -0 -P $(nproc) -I {} bash -c 'hipify-perl "{}" > "{}.hip" 2> "{}.warnings.txt"'

# Aggregate warnings for bulk analysis
cat **/*.warnings.txt | grep "warning:" | sort | uniq -c | sort -rn
```

---

## Scope: What This Skill Covers

### ✅ Automated (No Manual Effort)

This skill automates **basic CUDA-to-HIP conversion** for single-file projects using standard CUDA Runtime API:

| Coverage | HIPIFY % | AI Fixes | Iterations | Example Projects |
|----------|----------|----------|------------|------------------|
| Pure runtime API | 100% | 0-2 | 1 | vectorAdd, matrixMul, clock |
| SDK helper usage | 90-100% | 1-3 | 1-2 | asyncAPI, simpleStreams |
| Basic textures | 90-100% | 2-4 | 1-3 | simpleTexture, simpleLayeredTexture |

**Expected pass rate: ~90% for simple single-file CUDA samples**

### ⚠️ Requires Manual Effort (Outside Basic Skill)

These features require **expert manual intervention** - the skill provides guidance but not automation:

| Feature | HIPIFY % | Manual Effort | Time Estimate | Details |
|---------|----------|---------------|---------------|---------|
| **Multi-file projects** | 80-100% | Merge files, resolve duplicates | 1-4 hours | S16, S20 |
| **Cooperative groups (wave64)** | 90-100% | Fix warp size assumptions | 1-2 hours | S8, S18 |
| **cuBLAS/cuFFT libraries** | 70-90% | Link flags, API differences | 2-4 hours | S11, S26, S27 |
| **Complex textures** | 60-90% | Descriptor structs, filtering | 2-4 hours | S6, S22, S23 |

### 🚫 Blockers (Cannot Be Automated)

These features **cannot be converted** by this skill - require architectural changes or are not supported:

| Feature | HIPIFY % | Reason | Action Required |
|---------|----------|--------|-----------------|
| **Inline PTX assembly** | 0% | NVIDIA ISA incompatible with AMD | Rewrite in C++ or use AMD intrinsics |
| **WMMA/Tensor Cores** | 0% | Different matrix fragment API | Full rewrite using rocWMMA (~1-2 weeks) |
| **Dynamic Parallelism (CDP)** | 0% | Limited HIP support | Restructure to host-launched kernels |
| **OpenGL/Vulkan/D3D interop** | 0% | Not supported on Linux ROCm | Remove or use native HIP rendering |
| **NVRTC runtime compilation** | 0% | Different API | Port to HIPRTC (significant rewrite) |
| **cuDNN** | 0% | Use MIOpen instead | Full library port (~weeks) |

---

## Skills Reference (S1-S71)

Full tables with effort estimates and detection patterns are in `reference/api-rewrites.md`.

| Category | Skills | Where to look |
|----------|--------|---------------|
| Auto (script handles) | S1, S2, S4, S7, S28–S32, S40, S67–S71 | `reference/api-rewrites.md` |
| Partial (may need manual fixes) | S3, S6, S9, S14, S15, S19, S21, S33–S35, S39, S57–S62 | `reference/api-rewrites.md` |
| Manual / expert | S5, S8, S11–S13, S16–S18, S20, S22–S27, S36–S38, S42 | `reference/api-rewrites.md` |
| Blockers | S10, S24, S41, S43–S46, S52–S56 | `reference/api-rewrites.md` |
| PyTorch / MIOpen | S63–S66 | `reference/ml-inference-porting.md`, `tools/benchmark-template.py` |
| ONNX / TensorRT | S47 | `reference/ess-model-conversion.md` |
| Proactive fixes | S67–S71 | `reference/proactive-fixes.md` |

---

## Build Success Expectations

| Scenario | Success Rate |
|----------|-------------|
| hipify conversion completes | 95–100% of files |
| Standalone build (files with `main()`) | 5–10% |
| CMake multi-file build (portable files) | 40–50% |
| After proactive fixes (S67–S71) | 70–85% |

**Kernel-only files failing standalone build is NORMAL** — they need linking with a companion main.
**Graphics interop always fails** — GL/Vulkan/D3D are not portable to Linux ROCm.

Report separately: conversion success, portable file count, complete programs built, kernel-only count, blockers found.

---

## Quick Reference

**Common errors and fixes** — see `reference/troubleshooting.md` for full list.

| Error | Fix |
|-------|-----|
| `'checkCudaErrors' undeclared` | Add `#include "hip_helper.h"` (S40) |
| `cooperative_groups.h not found` | Use `hip/hip_cooperative_groups.h` (S68) |
| `undefined reference to 'main'` | Kernel-only file — normal; link with main (S29) |
| `hipProfilerStart` returns error | Comment out — deprecated ROCm 7.2+ (S70) |
| Wrong numerical results | Check hardcoded `32` → `warpSize` (S67) |

**API mappings** — see `reference/api-rewrites.md`. Key: cuda* → hip*, `cudaStreamWaitEvent` needs 3rd param `0`, `__shfl_sync` → `__shfl` (no mask).

**Library flags** — see `reference/build-commands.md`. Key: `-lhipblas`, `-lhipfft`, `-lhiprand`, `-lMIOpen`.

**Library includes** — Fix include paths after hipify:
- `#include <hipblas.h>` → `#include <hipblas/hipblas.h>`
- `#include <hipfft.h>` → `#include <hipfft/hipfft.h>`
- Add `-I/opt/rocm/include` to compile command

**ROCm versions** — Use latest official ROCm Docker (`rocm/dev-ubuntu-24.04:latest`). Always set `LD_LIBRARY_PATH=/opt/rocm/lib`.

---

## Reference Files

| File | Read when… |
|------|-----------|
| `reference/api-rewrites.md` | API errors, library linking, S33–S56 patterns |
| `reference/hip_helper.md` | hip_helper.h source — write to Common/hip_helper.h (S40) |
| `reference/proactive-fixes.md` | Full S67–S71 scripts, large-project parallelization |
| `reference/docker-setup.md` | Container setup, GPU passthrough, permission errors |
| `reference/build-commands.md` | hipcc/nvcc flags, CMake config, library linking |
| `reference/hipify-workflow.md` | hipify-perl vs hipify-clang |
| `reference/troubleshooting.md` | Build errors, runtime crashes, Isaac ROS/NITROS |
| `reference/ml-inference-porting.md` | ONNX Runtime → MIGraphX EP (S46), S63–S66 |
| `reference/ess-model-conversion.md` | TensorRT plugin decomposition (S47) |
| `scripts/onnx-to-pytorch-converter.py` | S63 implementation |
| `tools/benchmark-template.py` | S65 PyTorch benchmark |
