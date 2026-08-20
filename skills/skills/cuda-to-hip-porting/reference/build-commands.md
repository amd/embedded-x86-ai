<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Build Commands

hipcc and nvcc compilation with auto GPU detection.

---

## Auto-Detect GPU and Build

```bash
# Detect current GPU
GPU=$(amdgpu-arch 2>/dev/null | head -1)

# Build with detected GPU (or compile-only if none)
if [ -n "$GPU" ]; then
    echo "Building for $GPU"
    hipcc -I. --offload-arch=$GPU -w -o output output.hip
else
    echo "No GPU, compile-only"
    hipcc -I. -c -w -o output.o output.hip
fi
```

---

## Smart Recompile (check binary arch)

```bash
check_and_build() {
    local src="$1" bin="${1%.hip}_hip"
    local current=$(amdgpu-arch 2>/dev/null | head -1)
    
    [ -z "$current" ] && { hipcc -I. -c -w -o "${src%.hip}.o" "$src"; return; }
    
    local bin_arch=$(strings "$bin" 2>/dev/null | grep -oE 'gfx[0-9]+' | head -1)
    if [ ! -f "$bin" ] || [ "$bin_arch" != "$current" ]; then
        echo "Compiling for $current (was: ${bin_arch:-none})"
        hipcc -I. --offload-arch=$current -w -o "$bin" "$src"
    fi
    ./"$bin"
}

check_and_build myapp.hip
```

---

## hipcc Commands

```bash
# Basic with auto-detect
GPU=$(amdgpu-arch | head -1)
hipcc -O2 --offload-arch=$GPU -o output output.hip

# Compile-only (no GPU needed)
hipcc -O2 -c -o output.o output.hip

# Specific architectures
hipcc -O2 --offload-arch=gfx1151 -o output output.hip   # Strix Halo
hipcc -O2 --offload-arch=gfx90a  -o output output.hip   # MI200
hipcc -O2 --offload-arch=gfx942  -o output output.hip   # MI300X
hipcc -O2 --offload-arch=gfx1100 -o output output.hip   # RX 7900

# Multi-arch (runs on any listed GPU)
hipcc -O2 \
    --offload-arch=gfx90a \
    --offload-arch=gfx1100 \
    --offload-arch=gfx1151 \
    -o output output.hip

# With libraries
hipcc -O2 --offload-arch=$GPU -lhipblas -o output output.hip
hipcc -O2 --offload-arch=$GPU -lhipfft -o output output.hip
hipcc -O2 --offload-arch=$GPU -lhiprand -o output output.hip

# OpenMP support
hipcc -O2 --offload-arch=$GPU -fopenmp -o output output.hip

# Suppress warnings
hipcc -O2 --offload-arch=$GPU -w -o output output.hip
```

---

## nvcc Commands

```bash
# Basic
nvcc -O2 -std=c++17 -o output output.cu

# Specific architecture
nvcc -O2 -arch=sm_80 -o output output.cu   # A100
nvcc -O2 -arch=sm_90 -o output output.cu   # H100

# Multi-arch fat binary
nvcc -O2 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=sm_90 \
    -o output output.cu

# With libraries
nvcc -O2 -lcublas -lcublasLt -o output output.cu
nvcc -O2 -lcufft -o output output.cu
nvcc -O2 -lcurand -o output output.cu
```

---

## Architecture Mapping

| NVIDIA | AMD | Notes |
|--------|-----|-------|
| sm_70 (V100) | gfx906 (MI50) | First-gen tensor cores |
| sm_80 (A100) | gfx90a (MI200) | BF16 + FP64 |
| sm_86 (RTX 3090) | gfx1100 (RX 7900) | Consumer |
| sm_89 (RTX 4090) | gfx1100 | |
| sm_90 (H100) | gfx942 (MI300X) | FP8, wide matrix |
| Consumer APU | gfx1151 (Strix Halo) | RDNA3, ROCm 7.2+ |

---

## Library Link Flags

| CUDA | HIP | Link |
|------|-----|------|
| `-lcublas` | `-lhipblas` | BLAS |
| `-lcublasLt` | `-lhipblaslt` | BLAS Lt |
| `-lcusparse` | `-lhipsparse` | Sparse |
| `-lcufft` | `-lhipfft` | FFT |
| `-lcurand` | `-lhiprand` | Random |
| `-lcudnn` | `-lMIOpen` | DNN |
| Thrust | rocThrust | header-only |
| CUB | hipCUB | header-only |
| WMMA | rocWMMA | header-only |

---

## Flag Mapping

| nvcc | hipcc |
|------|-------|
| `--use_fast_math` | `-ffast-math` |
| `-Xcompiler -fopenmp` | `-fopenmp` |
| `-G` (debug) | `-ggdb` |
| `--threads=0` | (parallel by default) |
| `-lineinfo` | `-gline-tables-only` |

---

## Query GPU

```bash
# Get current GPU arch
amdgpu-arch              # Returns: gfx1151
rocminfo | grep -m1 gfx  # Alternative

# System info
rocm-smi
hipcc --version
rocminfo | grep "Marketing Name"
```

---

## CMake Gotchas

### Avoid find_package(HIP) - Use find_package(hip)

**Problem:** `find_package(HIP)` (uppercase) causes `-x hip` flag pollution across your entire build, breaking non-HIP source files.

```cmake
# BAD - causes -x hip flag pollution
find_package(HIP REQUIRED)
hip_add_executable(myapp main.cpp kernel.hip)

# GOOD - clean separation
find_package(hip REQUIRED)

# For pure HIP executables
add_executable(myapp main.cpp kernel.hip)
target_link_libraries(myapp hip::device)

# For mixed C++/HIP projects
set_source_files_properties(kernel.hip PROPERTIES LANGUAGE HIP)
add_executable(myapp main.cpp kernel.hip)
target_link_libraries(myapp hip::device)
```

**Symptoms of flag pollution:**
- C++ files getting compiled with `-x hip`
- Template errors in standard library headers
- Errors like "unknown CUDA architecture"

---

### LANGUAGE HIP Subdirectory Bug

**Problem:** `set_source_files_properties(... LANGUAGE HIP)` doesn't propagate to files in subdirectories when set in the parent CMakeLists.txt.

```cmake
# BAD - LANGUAGE HIP won't apply to subdirectory files
set_source_files_properties(
    subdir/kernel.hip
    PROPERTIES LANGUAGE HIP
)
add_subdirectory(subdir)

# GOOD - Set LANGUAGE HIP in the subdirectory's CMakeLists.txt
# In subdir/CMakeLists.txt:
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip
    PROPERTIES LANGUAGE HIP
)

# OR - Use absolute paths AFTER add_subdirectory
add_subdirectory(subdir)
get_target_property(SOURCES mylib SOURCES)
foreach(src ${SOURCES})
    if(src MATCHES "\\.hip$")
        set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
    endif()
endforeach()
```

**Root cause:** CMake file properties are scoped to the directory where they're set.

---

### Complete CMake Template for HIP Projects

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_hip_project LANGUAGES CXX HIP)

# Find HIP (lowercase!)
find_package(hip REQUIRED)

# Detect GPU architecture
execute_process(
    COMMAND amdgpu-arch
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT GPU_ARCH)
    set(GPU_ARCH "gfx1100")  # Default fallback
endif()

message(STATUS "Building for GPU: ${GPU_ARCH}")

# Set architecture
set(CMAKE_HIP_ARCHITECTURES ${GPU_ARCH})

# Create executable
add_executable(myapp
    main.cpp
    kernels.hip
)

# Mark HIP files explicitly (recommended for mixed projects)
set_source_files_properties(kernels.hip PROPERTIES LANGUAGE HIP)

# Link HIP runtime
target_link_libraries(myapp hip::device)

# Optional: HIP libraries
# target_link_libraries(myapp hipblas hiprand)
```

---

## Third-Party Library Strategy

### Pre-built Packages vs Source Builds

**Strongly prefer pre-built packages** when available:

| Library | Source Build Time | Package Solution |
|---------|-------------------|------------------|
| stdgpu | 15+ hours debugging | `apt install libstdgpu-hip-dev` (Debian trixie) |
| ONNX Runtime | 2-4 hours | `pip install onnxruntime-rocm` |
| PyTorch | 4-8 hours | `pip install torch --index-url .../rocm6.2` |

**When to build from source:**
- Package not available for your ROCm version
- Need specific compile-time options
- Debugging library internals

### stdgpu HIP Backend Fix

stdgpu (GPU hash tables) is commonly used by nvblox and similar libraries. Building with HIP backend requires CMake patches:

**The Problem:**
```
Building HIP object src/stdgpu/impl/device.cpp.o      ✅
Building CXX object src/stdgpu/hip/impl/device.cpp.o  ❌  # Should be HIP!
```

**Root Cause:** `set_source_files_properties(LANGUAGE HIP)` doesn't propagate from subdirectories.

**The Fix:**

```cmake
# In PARENT CMakeLists.txt (not in hip/ subdirectory):

# Add subdirectory first
add_subdirectory(src/stdgpu/hip)

# Then set LANGUAGE HIP with ABSOLUTE paths
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stdgpu/hip/impl/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stdgpu/hip/impl/memory.cpp
    PROPERTIES LANGUAGE HIP
)
```

**Alternative:** Use `target_sources()` with absolute paths in the subdirectory:

```cmake
# In hip/CMakeLists.txt:
target_sources(stdgpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/memory.cpp
)
```

### RAFT Dependency Strategy

Many ML libraries (cuML, cuGraph) depend on RAFT. **Build RAFT first:**

```bash
# 1. Clone and build RAFT with HIP
git clone https://github.com/rapidsai/raft
cd raft
# Apply HIP patches, build with hipcc

# 2. Then build dependent library
export RAFT_ROOT=/path/to/raft/install
cmake .. -DRAFT_ROOT=$RAFT_ROOT
```

**Blocking Pattern:**
```
fatal error: 'raft/core/handle.hpp' file not found
```

This means RAFT must be ported to HIP before the dependent library.
