<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# CUB to rocPRIM/hipCUB Conversion Guide

CUB (CUDA Unbound) primitives have good support via rocPRIM and hipCUB.

**Success rate:** 70-90% for established APIs, 0% for CCCL 3.3+ features

---

## Two Options

### Option 1: hipCUB (Recommended)

**CUB-compatible wrapper** around rocPRIM:
- Drop-in replacement for most CUB code
- Header paths same as CUB
- No code changes for basic usage

```cpp
// CUDA
#include <cub/device/device_reduce.cuh>

// HIP - same path, hipCUB provides compatibility
#include <cub/device/device_reduce.cuh>

// Link: header-only, no flags needed
```

### Option 2: rocPRIM (Direct)

**Native AMD implementation:**
- More control, potentially faster
- Different header paths
- May require code changes

```cpp
// CUDA
#include <cub/device/device_reduce.cuh>

// HIP - rocPRIM direct
#include <rocprim/device/device_reduce.hpp>

// API differences (see below)
```

**Recommendation:** Use hipCUB for faster porting, rocPRIM for optimization.

---

## API Compatibility Matrix

### Device-Level Algorithms (Fully Supported)

| Algorithm | CUB | hipCUB | rocPRIM | Notes |
|-----------|-----|--------|---------|-------|
| **DeviceReduce** | ✓ | ✓ | ✓ | All operations supported |
| `::Sum` | ✓ | ✓ | ✓ | Identical API |
| `::Min` | ✓ | ✓ | ✓ | Identical API |
| `::Max` | ✓ | ✓ | ✓ | Identical API |
| `::Reduce` (custom op) | ✓ | ✓ | ✓ | Identical API |
| **DeviceScan** | ✓ | ✓ | ✓ | All operations supported |
| `::InclusiveSum` | ✓ | ✓ | ✓ | Identical API |
| `::ExclusiveSum` | ✓ | ✓ | ✓ | Identical API |
| `::InclusiveScan` (custom op) | ✓ | ✓ | ✓ | Identical API |
| **DeviceSelect** | ✓ | ✓ | ✓ | Filter operations |
| `::Flagged` | ✓ | ✓ | ✓ | Identical API |
| `::If` | ✓ | ✓ | ✓ | Identical API |
| `::Unique` | ✓ | ✓ | ✓ | Identical API |
| **DeviceRadixSort** | ✓ | ✓ | ✓ | All sort variants |
| `::SortKeys` | ✓ | ✓ | ✓ | Identical API |
| `::SortPairs` | ✓ | ✓ | ✓ | Identical API |
| `::SortKeysDescending` | ✓ | ✓ | ✓ | Identical API |

### Block-Level Algorithms (Fully Supported)

| Algorithm | CUB | hipCUB | rocPRIM | Notes |
|-----------|-----|--------|---------|-------|
| `BlockReduce` | ✓ | ✓ | ✓ | All reduction ops |
| `BlockScan` | ✓ | ✓ | ✓ | All scan ops |
| `BlockLoad` | ✓ | ✓ | ✓ | Coalesced/striped loading |
| `BlockStore` | ✓ | ✓ | ✓ | Coalesced/striped storing |
| `BlockRadixSort` | ✓ | ✓ | ✓ | Block-level sort |

### CCCL 3.3+ Features (NOT Supported)

| Feature | CUB/CCCL 3.3+ | hipCUB/rocPRIM | Alternative |
|---------|---------------|----------------|-------------|
| `DeviceFind` | ✓ | ❌ | Manual search kernel |
| `DeviceSegmentedScan` | ✓ | ❌ | Use older `SegmentedReduce` |
| `DeviceTransform` (N→M) | ✓ | ❌ | Manual kernel |
| `cuda::std::identity` | ✓ | ❌ | Custom functor |
| `cuda::std::tuple` | ✓ | ❌ | Use struct |
| `cuda::std::functional` | ✓ | ❌ | Custom operator |

---

## Example 1: Device Reduce (Fully Compatible)

```cpp
// CUDA and HIP - identical
#include <cub/device/device_reduce.cuh>

void reduce_sum(float *d_in, float *d_out, int num_items) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temp storage size
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_in, d_out, num_items);

    // Allocate temp storage
    hipMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_in, d_out, num_items);

    hipFree(d_temp_storage);
}
```

**No changes needed - works identically on HIP.**

---

## Example 2: Device Scan (Fully Compatible)

```cpp
#include <cub/device/device_scan.cuh>

void prefix_sum(int *d_in, int *d_out, int num_items) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Inclusive sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                   d_in, d_out, num_items);

    hipMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                   d_in, d_out, num_items);

    hipFree(d_temp_storage);
}
```

**No changes needed.**

---

## Example 3: Device Select (Fully Compatible)

```cpp
#include <cub/device/device_select.cuh>

struct IsEven {
    __device__ bool operator()(int x) const { return x % 2 == 0; }
};

void select_even(int *d_in, int *d_out, int *d_num_selected, int num_items) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    IsEven predicate;

    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          d_in, d_out, d_num_selected, num_items, predicate);

    hipMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          d_in, d_out, d_num_selected, num_items, predicate);

    hipFree(d_temp_storage);
}
```

**No changes needed - custom predicates work identically.**

---

## Example 4: Radix Sort (Fully Compatible)

```cpp
#include <cub/device/device_radix_sort.cuh>

void sort_pairs(int *d_keys, int *d_values, int num_items) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Sort key-value pairs
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                     d_keys, d_keys,    // in-place for keys
                                     d_values, d_values, // in-place for values
                                     num_items);

    hipMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                     d_keys, d_keys,
                                     d_values, d_values,
                                     num_items);

    hipFree(d_temp_storage);
}
```

**No changes needed.**

---

## Example 5: Block-Level Reduce

```cpp
#include <cub/block/block_reduce.cuh>

__global__ void blockReduceKernel(float *input, float *output, int n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    float block_sum = BlockReduce(temp_storage).Sum(val);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = block_sum;
    }
}
```

**No changes needed - works identically on HIP.**

---

## CCCL 3.3+ Issues (Not Supported)

### Issue 1: cuda::std::identity

```cpp
// CUDA CCCL 3.3+
#include <cuda/std/functional>
cub::DeviceTransform::Apply(d_in, d_out, n, cuda::std::identity{});

// HIP - NOT SUPPORTED
// Use custom identity functor instead:
struct Identity {
    template<typename T>
    __device__ T operator()(const T& x) const { return x; }
};

// Manual transform kernel
__global__ void transform(float *out, const float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}
```

### Issue 2: cuda::std::tuple

```cpp
// CUDA CCCL 3.3+
using Tuple = cuda::std::tuple<int, float>;

// HIP - NOT SUPPORTED
// Use struct instead:
struct Pair {
    int first;
    float second;
};
```

### Issue 3: DeviceFind (CCCL 3.3+)

```cpp
// CUDA CCCL 3.3+
#include <cub/device/device_find.cuh>
cub::DeviceFind::FirstOf(d_temp, temp_bytes, d_in, d_out, n, predicate);

// HIP - NOT SUPPORTED
// Use manual search kernel:
__global__ void find_first(int *d_in, int *d_out, int n, Predicate pred) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && pred(d_in[idx])) {
        atomicMin(d_out, idx);  // First occurrence
    }
}
```

---

## rocPRIM Direct Usage

If using rocPRIM instead of hipCUB:

```cpp
// CUDA
#include <cub/device/device_reduce.cuh>
cub::DeviceReduce::Sum(...)

// HIP - rocPRIM direct
#include <rocprim/device/device_reduce.hpp>
rocprim::reduce(...)  // Note: different function name

// Parameter order may differ slightly
```

**Header path changes:**
- `cub/device/device_*.cuh` → `rocprim/device/device_*.hpp`
- `cub/block/block_*.cuh` → `rocprim/block/block_*.hpp`

**Recommendation:** Use hipCUB wrapper to avoid API differences.

---

## Compilation

### Using hipCUB (header-only)

```bash
# No special flags needed
hipcc -I. --offload-arch=gfx1151 -o sample sample.hip

# hipCUB headers included with ROCm
```

### Using rocPRIM (header-only)

```bash
# Same as hipCUB
hipcc -I. --offload-arch=gfx1151 -o sample sample.hip
```

---

## Troubleshooting

### Error: cub/device/device_*.cuh not found

```bash
# Install hipCUB if not present
# Usually included with ROCm, check:
ls /opt/rocm/include/hipcub
```

### Error: cuda::std::* not defined

```cpp
// CCCL 3.3+ features not in ROCm 7.2
// Use alternatives (struct, custom functor)
```

### Error: Different results than CUDA

Check for:
- Wave size assumptions (32 vs 64)
- Block size limits (check GPU specs)
- Numerical precision (use same data types)

---

## Performance Notes

### Temporary Storage

**Pattern:** Always query temp storage size first:

```cpp
void *d_temp = nullptr;
size_t temp_bytes = 0;

// Query size
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);

// Allocate
hipMalloc(&d_temp, temp_bytes);

// Run
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
```

**Why:** Temp storage size may differ between platforms.

### In-Place Operations

Many CUB operations support in-place:

```cpp
// In-place sort
cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
                                d_keys, d_keys, n);  // Same pointer
```

---

## Summary

**CUB to hipCUB: 70-90% success for established APIs**

✅ **Fully supported:**
- DeviceReduce (all ops)
- DeviceScan (all ops)
- DeviceSelect (all ops)
- DeviceRadixSort (all variants)
- BlockReduce, BlockScan, BlockLoad, BlockStore

❌ **Not supported:**
- CCCL 3.3+ features (DeviceFind, N→M Transform)
- `cuda::std::*` namespace utilities
- DeviceSegmentedScan (newer version)

🔧 **Migration steps:**
1. Use hipCUB wrapper (recommended) or rocPRIM direct
2. No code changes for basic CUB APIs
3. Replace CCCL 3.3+ features with custom kernels
4. Test temp storage allocation (may differ)

**Effort:** 3-4 hours per sample for basic CUB, weeks for CCCL 3.3+ features

**Result:** 70-90% compatibility for standard CUB patterns.
