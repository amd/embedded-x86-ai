# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# Skill Card

## Description

Ports NVIDIA CUDA codebases to AMD HIP/ROCm end-to-end. Covers 71 porting patterns (S1–S71): hipify-perl automation, proactive wave-size and cooperative-groups fixes (S67–S71), library mapping (cuBLAS→hipBLAS, cuFFT→hipFFT, cuDNN→MIOpen), Docker and ROCm environment setup per GPU arch (gfx1151 requires ROCm 7.2+), and ML inference migration (ONNX→MIGraphX, TensorRT plugin decomposition). Use when the user's request involves: hipify, CUDA-to-HIP, ROCm porting, .cu to .hip, hipcc, HIP migration, CUDAExtension, MIOpen tuning, gfx1151, or Strix Halo.

## Owner

AMD

## License

MIT
