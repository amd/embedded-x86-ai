<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# API Rewrites Reference

Complete API mappings and manual rewrites for CUDA to HIP conversion.

---

## Memory Management

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` |
| `cudaMemset` | `hipMemset` |
| `cudaMallocHost` | `hipHostMalloc` |
| `cudaFreeHost` | `hipHostFree` |
| `cudaMallocManaged` | `hipMallocManaged` |
| `cudaMallocAsync` | `hipMallocAsync` |
| `cudaFreeAsync` | `hipFreeAsync` |

---

## Streams & Events

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaStreamCreate` | `hipStreamCreate` | |
| `cudaStreamDestroy` | `hipStreamDestroy` | |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | |
| `cudaStreamWaitEvent(s, e)` | `hipStreamWaitEvent(s, e, 0)` | **Add 3rd param** |
| `cudaEventCreate` | `hipEventCreate` | |
| `cudaEventCreate(&e, f)` | `hipEventCreateWithFlags(&e, f)` | **Different name** |
| `cudaEventDestroy` | `hipEventDestroy` | |
| `cudaEventRecord` | `hipEventRecord` | |
| `cudaEventSynchronize` | `hipEventSynchronize` | |
| `cudaEventElapsedTime` | `hipEventElapsedTime` | |

---

## Device Management

| CUDA | HIP |
|------|-----|
| `cudaGetDeviceCount` | `hipGetDeviceCount` |
| `cudaSetDevice` | `hipSetDevice` |
| `cudaGetDevice` | `hipGetDevice` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaDeviceReset` | `hipDeviceReset` |
| `cudaDriverGetVersion` | `hipDriverGetVersion` |

---

## CUDA Graph API (S14)

| CUDA | HIP |
|------|-----|
| `cudaGraph_t` | `hipGraph_t` |
| `cudaGraphExec_t` | `hipGraphExec_t` |
| `cudaGraphCreate` | `hipGraphCreate` |
| `cudaGraphDestroy` | `hipGraphDestroy` |
| `cudaGraphInstantiate` | `hipGraphInstantiate` |
| `cudaGraphLaunch` | `hipGraphLaunch` |
| `cudaStreamBeginCapture` | `hipStreamBeginCapture` |
| `cudaStreamEndCapture` | `hipStreamEndCapture` |
| `cudaStreamCaptureModeGlobal` | `hipStreamCaptureModeGlobal` |
| `cudaGraphAddMemsetNode` | `hipGraphAddMemsetNode` |
| `cudaGraphAddKernelNode` | `hipGraphAddKernelNode` |

---

## API Signature Differences (S3)

```cpp
// hipStreamWaitEvent: 3 params (CUDA has 2)
cudaStreamWaitEvent(stream, event);        // CUDA
hipStreamWaitEvent(stream, event, 0);      // HIP

// hipMalloc3DArray: 4 params (CUDA has 3)
cudaMalloc3DArray(&arr, &desc, extent);         // CUDA
hipMalloc3DArray(&arr, &desc, extent, 0);       // HIP

// hipFuncGetAttributes: needs cast
cudaFuncGetAttributes(&attr, kernel);                    // CUDA
hipFuncGetAttributes(&attr, (const void*)kernel);        // HIP

// hipMemAdvise: int deviceId, not struct
cudaMemLocation loc; loc.type = ...; loc.id = deviceId;
cudaMemAdvise(ptr, size, advice, loc);          // CUDA (struct)
hipMemAdvise(ptr, size, advice, deviceId);      // HIP (int)
```

---

## Warp Intrinsics (S18)

```cpp
__shfl_sync(mask, val, src)      →  __shfl(val, src)
__shfl_down_sync(mask, val, d)   →  __shfl_down(val, d)
__shfl_up_sync(mask, val, d)     →  __shfl_up(val, d)
__shfl_xor_sync(mask, val, m)    →  __shfl_xor(val, m)
__ballot_sync(mask, pred)        →  __ballot(pred)
__any_sync(mask, pred)           →  __any(pred)
__all_sync(mask, pred)           →  __all(pred)
```

**Note:** `__ballot()` returns 64-bit on CDNA (64-thread wavefront).

---

## Vector Types (S19)

```cpp
// WRONG - hipcc errors on single-value
make_float4(0);
make_float2(3.0f);

// CORRECT - all components explicit
make_float4(0.0f, 0.0f, 0.0f, 0.0f);
make_float2(3.0f, 3.0f);
```

---

## Cache Control Intrinsics (S10)

HIP has no equivalents. Replace with plain memory access:

```cpp
__stcs(out + idx, val);   →  *(out + idx) = val;
__ldcs(in + idx);         →  *(in + idx);
__stcg(ptr, v);           →  *ptr = v;
```

---

## WMMA → rocWMMA (S25)

Hipify does NOT translate WMMA. Full manual rewrite required:

```cpp
// CUDA
#include <mma.h>
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::load_matrix_sync(a_frag, ptr, ldm);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// HIP (rocWMMA)
#include <rocwmma/rocwmma.hpp>
rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, _Float16, rocwmma::row_major> a_frag;
rocwmma::load_matrix_sync(a_frag, ptr, ldm);
rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**Block size:** CUDA warp=32, AMD CDNA wavefront=64. Adjust thread counts.

---

## cuBLASLt → hipBLASLt (S26)

| CUDA | HIP |
|------|-----|
| `cublasLtHandle_t` | `hipblasLtHandle_t` |
| `cublasLtCreate` | `hipblasLtCreate` |
| `cublasLtMatmulDesc_t` | `hipblasLtMatmulDesc_t` |
| `CUBLASLT_MATMUL_DESC_TRANSA` | `HIPBLASLT_MATMUL_DESC_TRANSA` |
| `CUBLASLT_MATMUL_DESC_SCALE_TYPE` | `HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE` |
| `-lcublasLt` | `-lhipblaslt` |

---

## PTX Assembly Replacement (S10)

| PTX | HIP Equivalent |
|-----|----------------|
| `dp4a` | Manual byte-wise dot product |
| `prmt.b32` | `__builtin_amdgcn_perm` |
| `vote.ballot` | `__ballot(pred)` |
| `mma.sync.*` | rocWMMA or MFMA builtins |
| `nanosleep` | No equivalent, remove |
| `cp.async.*` | `__builtin_amdgcn_global_load_lds` (gfx90a+) |

---

## cooperative_groups/reduce.h (S8)

No HIP equivalent. Implement manually:

```cpp
// CUDA
#include <cooperative_groups/reduce.h>
cg::reduce(tile, val, cg::plus<float>());

// HIP - manual warp reduction
float sum = val;
for (int i = warpSize/2; i > 0; i /= 2)
    sum += __shfl_down(sum, i);
```

---

## bfloat16 Conversions

```cpp
// CUDA
__nv_bfloat16 b = __float2bfloat16_rn(f);

// HIP (round-to-nearest is default)
__hip_bfloat16 b = __float2bfloat16(f);
```

---

## Common Missing Definitions

```cpp
// uint undefined
typedef unsigned int uint;

// EXIT_WAIVED undefined
#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// assert in device code
#include <cassert>

// NVTX stubs
#define nvtxRangePushA(x)
#define nvtxRangePop()
```

---

## Deprecated ROCm 7.2 APIs (S33)

### Profiler APIs

**Status:** Deprecated in ROCm 7.2 - always return `hipErrorNotSupported`

| CUDA API | HIP API (Deprecated) | Replacement |
|----------|---------------------|-------------|
| `cudaProfilerStart()` | `hipProfilerStart()` ❌ | roctracer, rocTX |
| `cudaProfilerStop()` | `hipProfilerStop()` ❌ | roctracer, rocTX |

**Fix pattern:**
```cpp
// Before (fails in ROCm 7.2+)
checkCudaErrors(hipProfilerStart());
// kernel launches
checkCudaErrors(hipProfilerStop());

// After (comment out with note)
// Note: hipProfilerStart/Stop deprecated in ROCm 7.2 (return hipErrorNotSupported)
// Use roctracer/rocTX for profiling instead
// checkCudaErrors(hipProfilerStart());
// kernel launches
// checkCudaErrors(hipProfilerStop());
```

---

## Error Code Compatibility (S34)

HIP error codes don't map 1:1 to CUDA. Tests that check specific error values may fail.

### Known Mismatches

| Error Type | CUDA Code | HIP Code | Notes |
|------------|-----------|----------|-------|
| Device assertion | `cudaErrorAssert` (59) | Error 719 | Different enumeration |
| Out of memory | `cudaErrorMemoryAllocation` | `hipErrorMemoryAllocation` | Usually maps correctly |
| Invalid value | `cudaErrorInvalidValue` | `hipErrorInvalidValue` | Usually maps correctly |

**Fix pattern:**
```cpp
// Before - checks specific error code
hipError_t err = hipGetLastError();
if (err == hipErrorAssert) {
    printf("Assertion failed\n");
}

// After - accept functional behavior
hipError_t err = hipGetLastError();
if (err != hipSuccess) {
    printf("Error occurred: %s (code %d)\n", hipGetErrorString(err), err);
    // Accept that assertions may return different codes on HIP
}
```

---

## APU vs Discrete GPU Patterns (S35)

### System-Wide Atomics

**Problem:** APU (integrated GPU) shares memory with CPU but has different coherence model. System-wide atomics can deadlock.

**Detection:**
```bash
# Find system-wide atomic functions
grep -r "atomic.*_system(" . --include="*.hip"

# Common patterns
atomicAdd_system(addr, val)
atomicInc_system(addr, val)
atomicDec_system(addr, val)
atomicCAS_system(addr, compare, val)
```

**Fix pattern:**

```cpp
// BEFORE - causes hangs on APU
__global__ void kernel(unsigned int *counter) {
    atomicInc_system(counter, UINT_MAX);
}

// CPU side
unsigned int *d_counter;
hipMalloc(&d_counter, sizeof(unsigned int));
kernel<<<1, 256>>>(d_counter);
unsigned int result;
hipMemcpy(&result, d_counter, sizeof(unsigned int), hipMemcpyDeviceToHost);

// AFTER - APU-safe version
__global__ void kernel(unsigned int *counter) {
    atomicInc(counter, UINT_MAX);  // Use device atomics
    __threadfence_system();         // Add system fence at end
}

// CPU side - force sync before accessing
unsigned int *d_counter;
hipMalloc(&d_counter, sizeof(unsigned int));
kernel<<<1, 256>>>(d_counter);
hipDeviceSynchronize();  // CRITICAL: force sync
unsigned int result;
hipMemcpy(&result, d_counter, sizeof(unsigned int), hipMemcpyDeviceToHost);

// CPU-side atomics - add retry limit
unsigned int expected, desired;
int retries = 0;
const int MAX_RETRIES = 1000;
do {
    expected = *d_counter;
    desired = (expected >= val) ? 0 : expected + 1;
    if (++retries > MAX_RETRIES) {
        fprintf(stderr, "CAS retry limit exceeded\n");
        break;
    }
} while (!__sync_bool_compare_and_swap(d_counter, expected, desired));
```

**Testing for hangs:**
```bash
# Use timeout to detect deadlocks
timeout 30 ./atomics_sample
if [ $? -eq 124 ]; then
    echo "TIMEOUT: Likely system atomics on APU"
fi
```

### Memory Coherence Differences

| Feature | Discrete GPU | APU (Integrated) |
|---------|--------------|------------------|
| System atomics | Usually safe | Can deadlock |
| Unified memory | Copies over PCIe | True shared memory |
| Synchronization | Explicit only | May need extra fences |

---

## Wave Size Portability (S39)

### The Problem

**NVIDIA:** 32 threads per warp
**AMD:** 64 threads per wave (CDNA/RDNA)

Hardcoding 32 causes silent numerical errors!

**Impact:** Test case showed 27x magnitude error from hardcoded wave size.

### Detection Patterns

```bash
# Find hardcoded 32 in warp operations
grep -rn "for.*offset.*32\|__shfl.*32\|warpSize.*32" . --include="*.hip"

# Common anti-patterns
for (int offset = 32 / 2; ...)     # BAD
for (int i = 0; i < 32; i++)       # BAD (if used for shuffles)
mask = (1ULL << 32) - 1            # BAD (should use warpSize)
```

### Fix Patterns

#### Warp Shuffle Reduction

```cpp
// BEFORE - hardcoded 32 (WRONG on AMD)
__device__ float warpReduceSum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// AFTER - portable (CORRECT)
__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}
```

#### Block Reduction

```cpp
// BEFORE - assumes 32-thread warps
__global__ void reduce(float *input, float *output, int n) {
    __shared__ float warp_sums[32];  // BAD: assumes max 32 warps

    float val = input[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {  // BAD: hardcoded
        val += __shfl_down(val, offset);
    }
    // ...
}

// AFTER - portable
__global__ void reduce(float *input, float *output, int n) {
    __shared__ float warp_sums[64];  // Support up to 64 warps (or use formula)

    float val = input[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    // ...
}
```

#### Lane Masks

```cpp
// BEFORE - 32-bit mask
unsigned int mask = (1U << 32) - 1;  // BAD: only 32 bits

// AFTER - 64-bit mask for AMD
unsigned long long mask = (1ULL << warpSize) - 1;  // CORRECT
```

### Architecture-Specific Code

If absolutely necessary to handle both:

```cpp
#if defined(__CUDA_ARCH__)
    const int WARP_SIZE = 32;
#elif defined(__HIP_DEVICE_COMPILE__)
    const int WARP_SIZE = warpSize;  // 64 on CDNA
#endif
```

**But prefer:** Just use `warpSize` built-in - works everywhere.

---

## CUDA Tile API Manual Conversion (S45)

### Overview

CUDA Tile API has NO direct HIP equivalent. Requires manual implementation using:
- Warp shuffle intrinsics
- Shared memory tiling
- Block-level reductions

### Warp-Level Reduction Template

```cpp
// CUDA Tile API (no HIP equivalent)
#include <cuda/tile>
namespace tile = cuda::cooperative_groups::tile;

// Manual HIP implementation
template<typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}
```

### Block-Level Reduction Template

```cpp
template<typename T>
__device__ T blockReduceSum(T val) {
    // Warp reduction
    val = warpReduceSum(val);

    // Inter-warp reduction via shared memory
    __shared__ T warp_sums[64];  // Max 64 warps per block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    // Final reduction
    val = (threadIdx.x < blockDim.x / warpSize) ? warp_sums[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}
```

### 2D Shared Memory Tiling (Matrix Transpose)

```cpp
// Optimal tile size (matches warp size)
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(float *out, const float *in, int width, int height) {
    // +1 to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Cooperative load into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write out transposed
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
```

**Key insights:**
- `TILE_DIM + 1` padding avoids shared memory bank conflicts
- Cooperative loading pattern is reusable
- Works identically on NVIDIA and AMD

---

## NPP Library Replacement Patterns (S43)

NPP (NVIDIA Performance Primitives) has no direct AMD equivalent. Use OpenCV with HIP backend or custom HIP kernels.

### NPP → OpenCV Function Mappings

| NPP Function | OpenCV Replacement | Notes |
|--------------|-------------------|-------|
| `nppiResize_8u_C1R` | `cv::resize()` | Use `INTER_LINEAR` for default |
| `nppiResize_8u_C3R` | `cv::resize()` | Works with 3-channel |
| `nppiRGBToGray_8u_C3C1R` | `cv::cvtColor(src, dst, COLOR_RGB2GRAY)` | |
| `nppiConvert_8u32f_C1R` | `src.convertTo(dst, CV_32F, 1.0/255.0)` | Scale optional |
| `nppiConvert_32f8u_C1R` | `src.convertTo(dst, CV_8U, 255.0)` | Scale to 0-255 |
| `nppiCopy_8u_C1R` | `src.copyTo(dst)` | Simple copy |
| `nppiYCbCr420ToRGB_8u_P3C3R` | `cv::cvtColor(src, dst, COLOR_YUV2RGB_I420)` | YUV420 → RGB |
| `nppiYCbCrToBGR_8u_C3R` | `cv::cvtColor(src, dst, COLOR_YCrCb2BGR)` | |
| `nppiFilterGauss_8u_C1R` | `cv::GaussianBlur()` | |
| `nppiFilterSobelHoriz_8u_C1R` | `cv::Sobel(src, dst, CV_8U, 1, 0)` | |
| `nppiFilterSobelVert_8u_C1R` | `cv::Sobel(src, dst, CV_8U, 0, 1)` | |
| `nppiThreshold_LTVal_8u_C1R` | `cv::threshold()` | Use `THRESH_BINARY` |
| `nppiMirror_8u_C1R` | `cv::flip()` | 0=vertical, 1=horiz |
| `nppiRotate_8u_C1R` | `cv::warpAffine()` | Build rotation matrix |
| `nppiHistogramEven_8u_C1R` | `cv::calcHist()` | |
| `nppiSum_8u_C1R` | `cv::sum()` | Returns Scalar |
| `nppiMean_8u_C1R` | `cv::mean()` | |
| `nppiMin_8u_C1R` | `cv::minMaxLoc()` | Get min only |
| `nppiMax_8u_C1R` | `cv::minMaxLoc()` | Get max only |

### Custom HIP Kernels for Performance-Critical Operations

For latency-sensitive operations, replace NPP with custom HIP kernels:

```cpp
// NPP nppiResize replacement - bilinear interpolation
__global__ void resizeBilinear_kernel(
    const unsigned char* src, int srcW, int srcH,
    unsigned char* dst, int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    float srcX = x * scaleX;
    float srcY = y * scaleY;

    int x0 = (int)srcX, y0 = (int)srcY;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);

    float dx = srcX - x0, dy = srcY - y0;

    float val = (1-dx)*(1-dy)*src[y0*srcW + x0] +
                dx*(1-dy)*src[y0*srcW + x1] +
                (1-dx)*dy*src[y1*srcW + x0] +
                dx*dy*src[y1*srcW + x1];

    dst[y*dstW + x] = (unsigned char)val;
}

// NPP nppiRGBToGray replacement
__global__ void rgbToGray_kernel(
    const unsigned char* rgb, unsigned char* gray,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    // ITU-R BT.601 coefficients
    int r = rgb[idx * 3 + 0];
    int g = rgb[idx * 3 + 1];
    int b = rgb[idx * 3 + 2];
    gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

// NPP nppiConvert_8u32f replacement
__global__ void convert8uTo32f_kernel(
    const unsigned char* src, float* dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = src[idx] / 255.0f;
}
```

### OpenCV with HIP Backend Setup

```cmake
# CMakeLists.txt
find_package(OpenCV REQUIRED)
find_package(hip REQUIRED)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    hip::device
)
```

```cpp
// Enable OpenCV HIP backend (ROCm 6.0+)
#include <opencv2/core.hpp>
#include <opencv2/core/hip.hpp>

// Use GpuMat for GPU operations
cv::cuda::GpuMat gpuSrc, gpuDst;
gpuSrc.upload(cpuMat);
cv::cuda::resize(gpuSrc, gpuDst, cv::Size(newW, newH));
gpuDst.download(cpuResult);
```

### Alternative: MIVisionX/rocAL

For comprehensive image processing pipelines:

```bash
# Install MIVisionX
apt-get install mivisionx

# Or build rocAL from source
git clone https://github.com/ROCmSoftwarePlatform/rocAL
cd rocAL && mkdir build && cd build
cmake .. && make -j$(nproc)
```

**MIVisionX provides:**
- vxResizeImage, vxColorConvert equivalents
- Batch processing optimizations
- Integration with ML inference pipelines

---

## hip_helper.h - Complete Source (S12)

Replaces NVIDIA's `helper_cuda.h`. Create this file in your project:

```cpp
#ifndef HIP_HELPER_H
#define HIP_HELPER_H

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <libgen.h>
#include <math.h>

// ============================================================================
// Error Checking
// ============================================================================

#ifndef HIP_CHECK
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

#define checkCudaErrors(call) HIP_CHECK(call)

inline void getLastCudaError(const char *msg) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error: %s: %s\n", msg, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Device Selection
// ============================================================================

inline int findCudaDevice(int argc, const char **argv) {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No HIP devices found!\n");
        exit(EXIT_FAILURE);
    }
    HIP_CHECK(hipSetDevice(0));
    return 0;
}

inline int findHipDevice(int argc, const char **argv) {
    return findCudaDevice(argc, argv);
}

inline int gpuDeviceInit(int devID) {
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No HIP devices found!\n");
        exit(EXIT_FAILURE);
    }
    if (devID < 0) devID = 0;
    if (devID >= device_count) devID = device_count - 1;
    HIP_CHECK(hipSetDevice(devID));
    return devID;
}

inline int gpuGetMaxGflopsDeviceId() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) return -1;
    int max_gflops_device = 0;
    unsigned long long max_gflops = 0;
    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        unsigned long long gflops = (unsigned long long)props.multiProcessorCount * props.clockRate;
        if (gflops > max_gflops) {
            max_gflops = gflops;
            max_gflops_device = i;
        }
    }
    return max_gflops_device;
}

inline bool checkCudaCapabilities(int major, int minor) {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    return deviceCount > 0;
}

inline int _ConvertSMVer2Cores(int major, int minor) {
    return 64;  // typical wavefront size
}

// ============================================================================
// Command Line Helpers
// ============================================================================

inline bool checkCmdLineFlag(int argc, const char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (strstr(argv[i], flag)) return true;
    return false;
}

inline int getCmdLineArgumentInt(int argc, const char **argv, const char *arg) {
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], arg)) {
            const char *val = strchr(argv[i], '=');
            if (val) return atoi(val + 1);
        }
    }
    return 0;
}

inline float getCmdLineArgumentFloat(int argc, const char **argv, const char *arg) {
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], arg)) {
            const char *val = strchr(argv[i], '=');
            if (val) return atof(val + 1);
        }
    }
    return 0.0f;
}

inline bool getCmdLineArgumentString(int argc, const char **argv, const char *arg, char **val) {
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], arg)) {
            const char *eq = strchr(argv[i], '=');
            if (eq) { *val = (char*)(eq + 1); return true; }
        }
    }
    return false;
}

// ============================================================================
// System Atomics Fallbacks
// ============================================================================

#ifndef atomicAdd_system
#define atomicAdd_system atomicAdd
#endif

#ifndef atomicInc_system
__device__ __forceinline__ unsigned int atomicInc_system(unsigned int *addr, unsigned int val) {
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, (assumed >= val) ? 0 : assumed + 1);
    } while (assumed != old);
    return old;
}
#endif

#ifndef atomicDec_system
__device__ __forceinline__ unsigned int atomicDec_system(unsigned int *addr, unsigned int val) {
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, ((assumed == 0) || (assumed > val)) ? val : assumed - 1);
    } while (assumed != old);
    return old;
}
#endif

// ============================================================================
// Shared Memory Helper
// ============================================================================

template <typename T>
class SharedMemory {
public:
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// ============================================================================
// Timer (SDK Compatible)
// ============================================================================

struct StopWatchInterface {
    struct timeval start_time, stop_time;
    float elapsed_ms;
    bool running;
};

inline void sdkCreateTimer(StopWatchInterface **timer) {
    *timer = (StopWatchInterface*)malloc(sizeof(StopWatchInterface));
    (*timer)->elapsed_ms = 0.0f;
    (*timer)->running = false;
}

inline void sdkDeleteTimer(StopWatchInterface **timer) {
    if (*timer) { free(*timer); *timer = NULL; }
}

inline void sdkStartTimer(StopWatchInterface **timer) {
    gettimeofday(&((*timer)->start_time), NULL);
    (*timer)->running = true;
}

inline void sdkStopTimer(StopWatchInterface **timer) {
    gettimeofday(&((*timer)->stop_time), NULL);
    (*timer)->running = false;
    long seconds = (*timer)->stop_time.tv_sec - (*timer)->start_time.tv_sec;
    long useconds = (*timer)->stop_time.tv_usec - (*timer)->start_time.tv_usec;
    (*timer)->elapsed_ms += (seconds * 1000.0f) + (useconds / 1000.0f);
}

inline void sdkResetTimer(StopWatchInterface **timer) {
    (*timer)->elapsed_ms = 0.0f;
    (*timer)->running = false;
}

inline float sdkGetTimerValue(StopWatchInterface **timer) {
    if ((*timer)->running) {
        struct timeval now;
        gettimeofday(&now, NULL);
        long seconds = now.tv_sec - (*timer)->start_time.tv_sec;
        long useconds = now.tv_usec - (*timer)->start_time.tv_usec;
        return (*timer)->elapsed_ms + (seconds * 1000.0f) + (useconds / 1000.0f);
    }
    return (*timer)->elapsed_ms;
}

inline float sdkGetAverageTimerValue(StopWatchInterface **timer) {
    return sdkGetTimerValue(timer);
}

// ============================================================================
// File I/O
// ============================================================================

template<typename T>
inline bool sdkWriteFile(const char *filename, T *data, unsigned int len, float epsilon, bool verbose) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    fwrite(data, sizeof(T), len, fp);
    fclose(fp);
    return true;
}

template<typename T>
inline bool sdkReadFile(const char *filename, T **data, unsigned int *len, bool verbose) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    *len = ftell(fp) / sizeof(T);
    fseek(fp, 0, SEEK_SET);
    *data = (T*)malloc(*len * sizeof(T));
    fread(*data, sizeof(T), *len, fp);
    fclose(fp);
    return true;
}

// ============================================================================
// PGM Image I/O
// ============================================================================

inline bool sdkLoadPGM(const char *filename, unsigned char **data, unsigned int *w, unsigned int *h) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) { fclose(fp); return false; }
    int c;
    while ((c = fgetc(fp)) == '#') while (fgetc(fp) != '\n');
    ungetc(c, fp);
    if (fscanf(fp, "%u %u", w, h) != 2) { fclose(fp); return false; }
    int maxval;
    if (fscanf(fp, "%d", &maxval) != 1) { fclose(fp); return false; }
    fgetc(fp);
    *data = (unsigned char*)malloc((*w) * (*h));
    if (fread(*data, 1, (*w) * (*h), fp) != (*w) * (*h)) { free(*data); fclose(fp); return false; }
    fclose(fp);
    return true;
}

inline bool sdkLoadPGM(const char *filename, float **data, unsigned int *w, unsigned int *h) {
    unsigned char *udata;
    if (!sdkLoadPGM(filename, &udata, w, h)) return false;
    *data = (float*)malloc((*w) * (*h) * sizeof(float));
    for (unsigned int i = 0; i < (*w) * (*h); i++) (*data)[i] = udata[i] / 255.0f;
    free(udata);
    return true;
}

inline bool sdkSavePGM(const char *filename, unsigned char *data, unsigned int w, unsigned int h) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    fprintf(fp, "P5\n%u %u\n255\n", w, h);
    fwrite(data, 1, w * h, fp);
    fclose(fp);
    return true;
}

inline bool sdkSavePGM(const char *filename, float *data, unsigned int w, unsigned int h) {
    unsigned char *udata = (unsigned char*)malloc(w * h);
    for (unsigned int i = 0; i < w * h; i++) udata[i] = (unsigned char)(data[i] * 255.0f);
    bool result = sdkSavePGM(filename, udata, w, h);
    free(udata);
    return result;
}

// ============================================================================
// PPM Image I/O (4-channel)
// ============================================================================

inline bool sdkLoadPPM4ub(const char *filename, unsigned char **data, unsigned int *w, unsigned int *h) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) { fclose(fp); return false; }
    int c;
    while ((c = fgetc(fp)) == '#') while (fgetc(fp) != '\n');
    ungetc(c, fp);
    if (fscanf(fp, "%u %u", w, h) != 2) { fclose(fp); return false; }
    int maxval;
    if (fscanf(fp, "%d", &maxval) != 1) { fclose(fp); return false; }
    fgetc(fp);
    unsigned char *rgb = (unsigned char*)malloc((*w) * (*h) * 3);
    if (fread(rgb, 1, (*w) * (*h) * 3, fp) != (*w) * (*h) * 3) { free(rgb); fclose(fp); return false; }
    *data = (unsigned char*)malloc((*w) * (*h) * 4);
    for (unsigned int i = 0; i < (*w) * (*h); i++) {
        (*data)[i*4+0] = rgb[i*3+0];
        (*data)[i*4+1] = rgb[i*3+1];
        (*data)[i*4+2] = rgb[i*3+2];
        (*data)[i*4+3] = 255;
    }
    free(rgb);
    fclose(fp);
    return true;
}

inline bool sdkSavePPM4ub(const char *filename, unsigned char *data, unsigned int w, unsigned int h) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    fprintf(fp, "P6\n%u %u\n255\n", w, h);
    for (unsigned int i = 0; i < w * h; i++) {
        fputc(data[i*4+0], fp);
        fputc(data[i*4+1], fp);
        fputc(data[i*4+2], fp);
    }
    fclose(fp);
    return true;
}

// ============================================================================
// File Path Helper
// ============================================================================

inline char* sdkFindFilePath(const char* filename, const char* argv0) {
    FILE *fp = fopen(filename, "r");
    if (fp) { fclose(fp); return strdup(filename); }
    char *path = strdup(argv0);
    char *dir = dirname(path);
    char *fullpath = (char*)malloc(strlen(dir) + strlen(filename) + 2);
    sprintf(fullpath, "%s/%s", dir, filename);
    free(path);
    return fullpath;
}

// ============================================================================
// Data Comparison
// ============================================================================

template<typename T>
inline bool compareData(const T *reference, const T *data, unsigned int len, T epsilon, float threshold) {
    unsigned int errors = 0;
    for (unsigned int i = 0; i < len; i++) {
        T diff = fabs(reference[i] - data[i]);
        if (diff > epsilon) errors++;
    }
    return (float)errors / len < threshold;
}

inline bool sdkCompareL2fe(const float *reference, const float *data, unsigned int len, float epsilon) {
    float error = 0.0f, ref = 0.0f;
    for (unsigned int i = 0; i < len; i++) {
        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }
    float normRef = sqrtf(ref);
    if (normRef < 1e-7f) return true;
    return (sqrtf(error) / normRef) < epsilon;
}

// ============================================================================
// Legacy Utilities
// ============================================================================

#define shrQAFinishExit(argc, argv, result) do { \
    printf("[%s] - %s\n", argv[0], (result) == 0 ? "Test PASSED" : "Test FAILED"); \
    exit((result) == 0 ? EXIT_SUCCESS : EXIT_FAILURE); \
} while(0)

#define shrQAFinish(argc, argv, result) shrQAFinishExit(argc, argv, result)

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#endif // HIP_HELPER_H
```

---

## S48: OpenCV CUDA → HIP Compatibility

**Discovered:** VSLAM porting (ORB-SLAM2-GPU, 2026-07-15)

OpenCV's GPU module uses `cv::cuda::*` types that hipify does NOT translate. These work with HIP via ROCm's OpenCV build but require a compatibility approach.

### Detection Patterns

```cpp
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/utility.hpp"
cv::cuda::GpuMat
cv::cuda::Stream
cv::cuda::StreamAccessor
using namespace cv::cuda;
```

### Solution: Compatibility Header

Create `opencv_hip_compat.hpp`:

```cpp
#ifndef OPENCV_HIP_COMPAT_HPP
#define OPENCV_HIP_COMPAT_HPP

#include <hip/hip_runtime.h>

// OpenCV CUDA headers work with HIP via ROCm OpenCV build
// The cv::cuda:: namespace is preserved - it's OpenCV's abstraction
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

#endif
```

### Key Points

- `cv::cuda::GpuMat` works with HIP memory via ROCm OpenCV
- `cv::cuda::Stream` can wrap `hipStream_t` via `StreamAccessor::wrapStream()`
- Keep `cv::cuda::` namespace - it's OpenCV's abstraction, not CUDA-specific
- Build OpenCV with ROCm backend: `-DWITH_HIP=ON`

### Example: Stream Wrapping

```cpp
// Works with both CUDA and HIP when OpenCV is built with ROCm
hipStream_t stream;
hipStreamCreate(&stream);
cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
```

---

## S50: hipMemcpyToSymbol HIP_SYMBOL Wrapper

**Discovered:** VSLAM porting (2026-07-15)

HIP requires `HIP_SYMBOL()` wrapper for `__constant__` memory variables in `hipMemcpyToSymbol`.

### CUDA

```cpp
__constant__ float c_data[256];
cudaMemcpyToSymbol(c_data, host_data, sizeof(float) * 256);
```

### HIP

```cpp
__constant__ float c_data[256];
hipMemcpyToSymbol(HIP_SYMBOL(c_data), host_data, sizeof(float) * 256);
```

### Note

hipify-perl usually handles this automatically, but verify in complex cases:
- Nested namespaces: `namespace A { namespace B { __constant__ ... } }`
- Template instantiation: `hipMemcpyToSymbol(HIP_SYMBOL(c_table<T>), ...)`
- Array of structs: `hipMemcpyToSymbol(HIP_SYMBOL(c_params[0]), ...)`

---

## S51: NVTX → ROCTX Profiling Annotations

**Discovered:** VSLAM porting (2026-07-15)

NVIDIA's NVTX profiling annotations need manual replacement with AMD's ROCTX.

### Detection Patterns

```cpp
#include <nvtx3/nvToolsExt.h>
nvtxRangePush("kernel");
nvtxRangePop();
PUSH_RANGE("name", color);
POP_RANGE;
```

### Solution

```cpp
#include <roctracer/roctx.h>

// Direct API usage
roctxRangePush("kernel");
roctxRangePop();

// Or create compatibility macros
#define PUSH_RANGE(name, color) roctxRangePush(name)
#define POP_RANGE roctxRangePop()
```

### API Mapping

| NVTX | ROCTX |
|------|-------|
| `nvtxRangePush(name)` | `roctxRangePush(name)` |
| `nvtxRangePop()` | `roctxRangePop()` |
| `nvtxRangePushA(name)` | `roctxRangePush(name)` |
| `nvtxMark(name)` | `roctxMark(name)` |

### Build

Replace `-lnvToolsExt` with `-lroctx64` in link flags.

### Note

- Color parameter is ignored in ROCTX - use string names for identification
- ROCTX ranges appear in `rocprof` output: `rocprof --roctx-trace ./myapp`

---

## S52: OpenCV PtrStep Adaptation

**Discovered:** cuVSLAM porting (2026-07-15)

OpenCV CUDA kernels use `PtrStepSzb` and `PtrStep<T>` types that require adaptation for cross-framework integration.

### Detection Patterns

```cpp
cv::cuda::PtrStepSzb
cv::cuda::PtrStep<float>
PtrStepSzb src
src.step, src.cols, src.rows
```

### Solution

Convert to raw pointer interface:

```cpp
// BEFORE (OpenCV PtrStep)
__global__ void kernel(PtrStepSzb src, PtrStepSzb dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < src.cols && y < src.rows) {
        dst.ptr(y)[x] = src.ptr(y)[x];
    }
}

// AFTER (raw pointer)
__global__ void kernel(const uchar* __restrict__ src, uchar* __restrict__ dst,
                       int width, int height, int src_step, int dst_step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        dst[y * dst_step + x] = src[y * src_step + x];
    }
}
```

### Effort

Manual: 2-4 hours per kernel file

---

## S53: Keypoint Structure Mapping

**Discovered:** cuVSLAM porting (2026-07-15)

Different frameworks use different keypoint representations. cuVSLAM uses `float2` arrays while rocSLAM uses separate `float*` arrays.

### Detection Patterns

```cpp
float2* keypoints
keypoints[i].x, keypoints[i].y
cv::KeyPoint kp
kp.pt.x, kp.pt.y
```

### Solution

```cpp
// BEFORE (float2 array)
__global__ void trackKernel(const float2* prev_kp, float2* curr_kp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = prev_kp[idx].x;
        float y = prev_kp[idx].y;
        // ...
        curr_kp[idx] = make_float2(new_x, new_y);
    }
}

// AFTER (separate arrays)
__global__ void trackKernel(const float* prev_x, const float* prev_y,
                            float* curr_x, float* curr_y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = prev_x[idx];
        float y = prev_y[idx];
        // ...
        curr_x[idx] = new_x;
        curr_y[idx] = new_y;
    }
}
```

### Effort

Manual: 1-2 hours per kernel

---

## S54: Framework Data Flow Adaptation

**Discovered:** cuVSLAM porting (2026-07-15)

Different SLAM frameworks have different data flow patterns. Adapting kernel I/O to match target framework conventions.

### Common Patterns

| Source (cuVSLAM) | Target (rocSLAM) |
|------------------|------------------|
| Single fused kernel | Separate gradient X/Y kernels |
| Texture object input | Raw pointer input |
| Pitched output | Row-major output |
| Struct-of-arrays | Array-of-structs |

### Solution

Create wrapper functions that adapt between calling conventions:

```cpp
// cuVSLAM style (single call, texture, pitched)
hipError_t conv_grad_xy(hipTextureObject_t src, uint2 srcSize,
                         float* gx, float* gy, size_t pitch, hipStream_t s);

// Wrapper for rocSLAM style (separate calls, raw pointer, contiguous)
hipError_t launchGradients_adapted(const float* d_img, float* d_gx, float* d_gy,
                                    int W, int H, hipStream_t stream) {
    // Call internal cuVSLAM kernels with adapted parameters
    gradXKernel_cuVSLAM<<<blocks, threads, 0, stream>>>(d_img, d_gx, W, H);
    gradYKernel_cuVSLAM<<<blocks, threads, 0, stream>>>(d_img, d_gy, W, H);
    return hipGetLastError();
}
```

### Effort

Manual: 2-4 hours per pipeline

---

## S55: CUDA Pipeline Primitives Replacement

**Discovered:** cuVSLAM porting (2026-07-15)

CUDA provides `cuda_pipeline_primitives.h` for hardware-accelerated async memory copy operations not available in HIP.

### Detection Patterns

```cpp
#include <cuda_pipeline_primitives.h>
__pipeline_memcpy_async(dst, src, size);
__pipeline_commit();
__pipeline_wait_prior(0);
```

### Solution Options

**Option A: hipMemcpyAsync (global→global)**
```cpp
hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
hipStreamSynchronize(stream);
```

**Option B: Cooperative groups memcpy_async (ROCm 6.0+)**
```cpp
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(...) {
    auto group = cg::this_thread_block();
    cg::memcpy_async(group, shared_dst, global_src, size);
    cg::wait(group);
}
```

**Option C: Manual copy (fallback)**
```cpp
__global__ void kernel(...) {
    for (int i = threadIdx.x; i < size/sizeof(float); i += blockDim.x) {
        shared_dst[i] = global_src[i];
    }
    __syncthreads();
}
```

### Affected Files

- `sba_imu_v1.hip` (line 23)

### Effort

Manual: 30 min - 2 hours depending on complexity

---

## S56: API Adaptation Layer (Cross-Framework Integration)

**Discovered:** cuVSLAM to rocSLAM integration (2026-07-15)

When porting kernels from one GPU library to integrate with another framework that has incompatible API conventions.

### Problem

| Aspect | cuVSLAM | rocSLAM |
|--------|---------|---------|
| Memory | Texture objects | Raw pointers |
| Layout | Pitched | Row-major contiguous |
| Size | `uint2` struct | Separate `int W, H` |
| Namespace | `cuvslam::hip` | `rocslam::tracking::hip` |

### Detection

- Source kernels use `hipTextureObject_t`
- Target framework expects `const float* __restrict__`
- Different parameter conventions between libraries

### Adaptation Steps

1. **Namespace change**: `cuvslam::hip` → `rocslam::tracking::hip::cuvslam_adapted`

2. **Texture → Raw Pointer**:
```cpp
// BEFORE
__global__ void kernel(hipTextureObject_t src, ...) {
    float val = tex2D<float>(src, x, y);
}

// AFTER
__global__ void kernel(const float* __restrict__ src, int width, ...) {
    float val = src[y * width + x];
}
```

3. **Pitched → Contiguous**:
```cpp
// BEFORE
float* dst_row = (float*)((char*)dst + y * dpitch) + x;

// AFTER
float* dst_row = dst + y * width + x;
```

4. **uint2 → Separate Params**:
```cpp
// BEFORE
void func(uint2 size) { ... size.x ... size.y ... }

// AFTER
void func(int W, int H) { ... W ... H ... }
```

### Integration Pattern

```cpp
#ifdef USE_CUVSLAM_KERNELS
  #include "cuvslam_adapted_kernels.hip"
  cuvslam_adapted::launchGradients_cuVSLAM(d_img, d_gx, d_gy, W, H, stream);
#else
  launchGradients(d_img, d_gx, d_gy, W, H, stream);
#endif
```

### Benchmark Results (KITTI Sequence 01)

| Kernel | rocSLAM Native | cuVSLAM (HIP port) | % Change ↑↓ |
|--------|---------------:|-------------------:|------------:|
| Speed | 79.6 s | 91.8 s | -15.3% |
| FPS | 13.8 fps | 12.0 fps | -13.0% |

*cuVSLAM (HIP port) is baseline. Positive = better.*

Note: rocSLAM is faster due to simpler gradient kernel (3 ops vs 14 ops).

### Effort

AI-assisted: ~1 day for full kernel file adaptation
