<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# cuda-to-hip-porting: Sample Project Types — What Each Exercises

This guide describes **types** of CUDA projects ordered from trivial to production-shaped,
as used when testing the `cuda-to-hip-porting` skill.
Use NVIDIA CUDA Samples (https://github.com/NVIDIA/cuda-samples) or your own projects.

| Type | Example | NVIDIA libs | CUDA features | hipify auto? | Manual rewrite? | Skills |
|------|---------|-------------|---------------|--------------|-----------------|--------|
| **Trivial** | vectorAdd | none | runtime API only | 100% | none | S1, S2 |
| **Simple** | matrixMul | none | shared memory | 100% | none | S1, S2 |
| **Library** | BLAS samples | cuBLAS | `__shfl_down_sync(0xFFFFFFFF, ...)` | mostly | wave-size hardcode | S1, S2, S11, S18 |
| **Wave64** | warp intrinsics | none | `__ballot_sync`, `cg::tiled_partition<32>` | partial | warp-size + cooperative-groups | S1, S2, S8, S18 |
| **Tensor Core** | WMMA samples | none | `nvcuda::wmma::fragment`, `mma_sync` | **zero** | full rewrite to `rocwmma::` | S1, S2, S25 |
| **PTX** | inline asm | none | `asm("dp4a.s32.s32 ...")` | partial | manual PTX rewrite | S1, S2, S10 |
| **Production** | LLM training | cuBLASLt, cuDNN, NCCL | everything | varies | many | S1-S56 |

## Why the order matters

- **Trivial/Simple** confirm the toolchain works before any complications.
- **Library** samples introduce cuBLAS → hipBLAS and the first hidden gotcha:
  the `0xFFFFFFFF` warp mask is a wave32 assumption that breaks on CDNA wave64.
- **Wave64** samples stack wave-size assumptions and cooperative groups.
  Most instructive for understanding S8 and S18.
- **Tensor Core** samples are where hipify cannot meaningfully translate.
  WMMA code requires full manual rewrite to rocWMMA.
- **PTX** samples demonstrate inline assembly issues (`dp4a` has a clean AMD intrinsic,
  but `mma.sync`, `cp.async` require significant rewrite).
- **Production** codebases are the real challenge: multiple files, many libraries,
  mixed precision, multi-GPU. Run the profiler first to see the full inventory.

## What the samples do NOT cover

Known blind spots in this corpus, worth supplementing later:

- **CUDA Graphs** (S14) - none of the samples use them. Real inference
  servers do.
- **Driver API** beyond NVML - llmc uses NVML for FLOPS reporting
  but no `cuCtx*` / `cuMemMap` / VMM patterns.
- **Atomics with explicit scope** (S7) - no sample exercises
  `cuda::atomic` or `__threadfence_system`.
- **CUDA Dynamic Parallelism** (S24) - rare in production; documented as a
  restructuring task, not a translation.

If you need to teach or test against those gaps, add a sample.

## Using NVIDIA CUDA Samples

Download from https://github.com/NVIDIA/cuda-samples (~190 individual samples).
Use as a feature-coverage corpus:

- **Pick by feature.** Need CUDA Graphs? Look in `Samples/3_CUDA_Features/`.
  Need NVRTC? Check `clock_nvrtc`, `matrixMul_nvrtc`. Need Driver API? `matrixMulDrv`.
- **Use the profiler to triage.** Run
  `python3 skills/cuda-to-hip-porting/scripts/profile_cuda_project.py /path/to/sample`
  on any sample to see what it exercises before porting.
- **Don't port everything.** Many samples depend on NVIDIA-only APIs
  (NVENC, NVDEC, OptiX, OpenGL interop) that have no AMD equivalent.

---

## Test Results

For current test results, see `skills/cuda-to-hip-porting/examples/cuml/results/cuml_comprehensive_report.md`.

### Known Blockers

| Blocker | Skills Needed |
|---------|---------------|
| OpenGL/EGL/Vulkan interop | N/A (no ROCm equivalent) |
| WMMA/mma.h | S25 (rocWMMA manual rewrite) |
| cooperative_groups/reduce.h | Manual warp reduction |
| Shared memory IPC | N/A |
| Inline PTX assembly | S10 (manual rewrite) |

### Common Compile Failures

| Reason | Fix |
|--------|-----|
| Multi-file redefinitions | S16, S20 |
| Missing hip_helper.h | S12 |
| cuda_runtime.h not found | S2 |
| API signature mismatch | S3 |
