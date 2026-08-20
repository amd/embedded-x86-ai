<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Thrust to rocThrust Conversion Guide

**Success rate: 90-95% for established Thrust algorithms**

rocThrust provides excellent compatibility for standard Thrust operations. CCCL 3.3+ libcu++ features are not supported.

---

## Header Paths (Usually No Change)

```cpp
// CUDA
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

// HIP - same paths
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
```

**rocThrust uses same header paths as Thrust** - no changes needed in most cases.

---

## API Compatibility

### Fully Supported STL-Like Algorithms (100% compatible)

| Algorithm | Thrust | rocThrust | Notes |
|-----------|--------|-----------|-------|
| **Sorting** |  |  |  |
| `thrust::sort` | ✓ | ✓ | Identical API |
| `thrust::stable_sort` | ✓ | ✓ | Identical API |
| `thrust::sort_by_key` | ✓ | ✓ | Identical API |
| **Reductions** |  |  |  |
| `thrust::reduce` | ✓ | ✓ | Identical API |
| `thrust::min_element` | ✓ | ✓ | Identical API |
| `thrust::max_element` | ✓ | ✓ | Identical API |
| **Scans** |  |  |  |
| `thrust::inclusive_scan` | ✓ | ✓ | Identical API |
| `thrust::exclusive_scan` | ✓ | ✓ | Identical API |
| **Transformations** |  |  |  |
| `thrust::transform` | ✓ | ✓ | Identical API |
| `thrust::transform_reduce` | ✓ | ✓ | Identical API |
| **Copy** |  |  |  |
| `thrust::copy` | ✓ | ✓ | Identical API |
| `thrust::copy_if` | ✓ | ✓ | Identical API |
| **Search** |  |  |  |
| `thrust::find` | ✓ | ✓ | Identical API |
| `thrust::find_if` | ✓ | ✓ | Identical API |
| `thrust::count` | ✓ | ✓ | Identical API |
| `thrust::count_if` | ✓ | ✓ | Identical API |
| **Unique** |  |  |  |
| `thrust::unique` | ✓ | ✓ | Identical API |
| `thrust::unique_by_key` | ✓ | ✓ | Identical API |
| **Fill** |  |  |  |
| `thrust::fill` | ✓ | ✓ | Identical API |
| `thrust::sequence` | ✓ | ✓ | Identical API |

### Container Types (Fully Supported)

| Type | Thrust | rocThrust | Notes |
|------|--------|-----------|-------|
| `device_vector<T>` | ✓ | ✓ | Identical API |
| `host_vector<T>` | ✓ | ✓ | Identical API |
| `device_ptr<T>` | ✓ | ✓ | Identical API |

### Iterators (Fully Supported)

| Iterator | Thrust | rocThrust | Notes |
|----------|--------|-----------|-------|
| `counting_iterator` | ✓ | ✓ | Identical API |
| `constant_iterator` | ✓ | ✓ | Identical API |
| `transform_iterator` | ✓ | ✓ | Identical API |
| `zip_iterator` | ✓ | ✓ | Identical API |
| `permutation_iterator` | ✓ | ✓ | Identical API |

### CCCL 3.3+ Features (NOT Supported)

| Feature | Thrust/CCCL 3.3+ | rocThrust | Alternative |
|---------|------------------|-----------|-------------|
| `cuda::std::identity` | ✓ | ❌ | Custom functor |
| `cuda::std::tuple` | ✓ | ❌ | Use `thrust::tuple` |
| `cuda::std::functional` | ✓ | ❌ | Custom operator |
| `cuda::std::optional` | ✓ | ❌ | Use sentinel value |

---

## Example 1: Device Vector (Fully Compatible)

```cpp
// CUDA and HIP - identical
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void vector_example() {
    // Host vector
    thrust::host_vector<float> h_vec(100);
    for (int i = 0; i < 100; i++) {
        h_vec[i] = i * 2.0f;
    }

    // Copy to device
    thrust::device_vector<float> d_vec = h_vec;

    // Access size
    size_t size = d_vec.size();

    // Copy back to host
    h_vec = d_vec;
}
```

**No changes needed - works identically on HIP.**

---

## Example 2: Sort (Fully Compatible)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void sort_example() {
    thrust::device_vector<int> d_vec(1000);

    // Fill with random values
    thrust::sequence(d_vec.begin(), d_vec.end(), 1000, -1);

    // Sort ascending
    thrust::sort(d_vec.begin(), d_vec.end());

    // Sort descending
    thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());
}
```

**No changes needed.**

---

## Example 3: Reduce (Fully Compatible)

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

void reduce_example() {
    thrust::device_vector<float> d_vec(1000, 1.0f);

    // Sum all elements
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    // Find minimum
    float min_val = thrust::reduce(d_vec.begin(), d_vec.end(), FLT_MAX, thrust::minimum<float>());

    // Find maximum
    float max_val = thrust::reduce(d_vec.begin(), d_vec.end(), -FLT_MAX, thrust::maximum<float>());
}
```

**No changes needed.**

---

## Example 4: Transform (Fully Compatible)

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>

struct square_functor {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

void transform_example() {
    thrust::device_vector<float> d_in(1000);
    thrust::device_vector<float> d_out(1000);

    thrust::sequence(d_in.begin(), d_in.end());

    // Square all elements
    thrust::transform(d_in.begin(), d_in.end(), d_out.begin(), square_functor());
}
```

**No changes needed - custom functors work identically.**

---

## Example 5: Scan (Fully Compatible)

```cpp
#include <thrust/device_vector.h>
#include <thrust/scan.h>

void scan_example() {
    thrust::device_vector<int> d_in(100);
    thrust::device_vector<int> d_out(100);

    thrust::sequence(d_in.begin(), d_in.end(), 1);

    // Inclusive prefix sum
    thrust::inclusive_scan(d_in.begin(), d_in.end(), d_out.begin());

    // Exclusive prefix sum
    thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());
}
```

**No changes needed.**

---

## Example 6: Custom Binary Operator

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

struct max_abs_functor {
    __host__ __device__
    float operator()(float a, float b) const {
        return fmaxf(fabsf(a), fabsf(b));
    }
};

void custom_reduce() {
    thrust::device_vector<float> d_vec(1000);

    // Reduce with custom operator
    float max_abs = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, max_abs_functor());
}
```

**No changes needed.**

---

## Example 7: Zip Iterator

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

void zip_sort() {
    thrust::device_vector<int> keys(100);
    thrust::device_vector<float> values(100);

    // Sort keys and values together
    thrust::sort(
        thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(keys.end(), values.end()))
    );
}
```

**No changes needed - zip iterators work identically.**

---

## Example 8: Transform Iterator

```cpp
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

struct negate_functor {
    __host__ __device__
    float operator()(float x) const { return -x; }
};

void transform_iterator_example() {
    thrust::device_vector<float> d_vec(100);

    // Reduce negated values without creating temp array
    auto neg_begin = thrust::make_transform_iterator(d_vec.begin(), negate_functor());
    auto neg_end = thrust::make_transform_iterator(d_vec.end(), negate_functor());

    float sum = thrust::reduce(neg_begin, neg_end);
}
```

**No changes needed.**

---

## Example 9: Counting Iterator

```cpp
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

void counting_iterator_example() {
    thrust::device_vector<int> d_out(100);

    // Generate sequence using counting iterator
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(100),
        d_out.begin(),
        thrust::placeholders::_1 * 2  // Multiply by 2
    );
}
```

**Works on HIP - use `thrust::placeholders` for lambdas.**

---

## CCCL 3.3+ Issues (Not Supported)

### Issue 1: cuda::std::identity

```cpp
// CUDA CCCL 3.3+
#include <cuda/std/functional>
thrust::transform(in.begin(), in.end(), out.begin(), cuda::std::identity{});

// HIP - NOT SUPPORTED
// Use custom identity functor:
struct identity {
    template<typename T>
    __host__ __device__
    T operator()(const T& x) const { return x; }
};

thrust::transform(in.begin(), in.end(), out.begin(), identity());
```

### Issue 2: cuda::std::tuple

```cpp
// CUDA CCCL 3.3+
using Tuple = cuda::std::tuple<int, float>;

// HIP - Use thrust::tuple instead
using Tuple = thrust::tuple<int, float>;
```

**thrust::tuple is supported in rocThrust.**

### Issue 3: cuda::std::optional

```cpp
// CUDA CCCL 3.3+
cuda::std::optional<int> opt;

// HIP - NOT SUPPORTED
// Use sentinel value instead:
const int NO_VALUE = -1;
int value = NO_VALUE;
```

---

## Compilation

```bash
# No special flags needed
hipcc -I. --offload-arch=gfx1151 -o sample sample.hip

# Thrust/rocThrust is header-only
```

---

## Performance Notes

### Execution Policies

Thrust automatically uses GPU for `device_vector`:

```cpp
thrust::device_vector<int> d_vec(100);

// Runs on GPU automatically
thrust::sort(d_vec.begin(), d_vec.end());

// Force host execution
thrust::host_vector<int> h_vec(100);
thrust::sort(h_vec.begin(), h_vec.end());  // Runs on CPU
```

### Temporary Storage

Thrust manages temporary storage automatically:

```cpp
// No manual temp storage needed
thrust::sort(d_vec.begin(), d_vec.end());
```

**Unlike CUB** - Thrust handles memory allocation internally.

---

## Troubleshooting

### Error: thrust headers not found

```bash
# Check ROCm installation
ls /opt/rocm/include/thrust

# Should contain rocThrust headers
```

### Error: Different results than CUDA

Check for:
- Numerical precision (float vs double)
- Stable vs unstable sort (use `thrust::stable_sort` if order matters)
- Custom comparator definitions

### Performance Differences

rocThrust may have different performance characteristics:
- Benchmark critical sections
- Consider tuning block sizes
- Profile with rocprof

---

## Lambda Functions

### Thrust Placeholders (Supported)

```cpp
#include <thrust/functional.h>

using namespace thrust::placeholders;

// Lambda-style using placeholders
thrust::transform(in.begin(), in.end(), out.begin(),
                  _1 * 2 + 5);  // x * 2 + 5
```

**Works on both CUDA and HIP.**

### Device Lambdas (Limited Support)

```cpp
// CUDA (C++14 device lambdas)
thrust::transform(in.begin(), in.end(), out.begin(),
                  [] __device__ (float x) { return x * 2; });

// HIP - may have limited support
// Use functors for portability
```

**Recommendation:** Use functors for cross-platform compatibility.

---

## Raw Pointer Interop

Thrust works with raw pointers:

```cpp
float *d_raw;
hipMalloc(&d_raw, n * sizeof(float));

// Wrap in device_ptr
thrust::device_ptr<float> d_ptr(d_raw);

// Use with Thrust algorithms
thrust::sort(d_ptr, d_ptr + n);

// Or use raw pointers directly
thrust::sort(thrust::device, d_raw, d_raw + n);
```

---

## Summary

**Thrust to rocThrust: 90-95% success for established algorithms**

✅ **Fully supported:**
- All STL-like algorithms (sort, reduce, scan, transform, copy, find, count, unique, fill)
- Containers (device_vector, host_vector)
- Iterators (counting, constant, transform, zip, permutation)
- Custom functors and operators
- thrust::tuple

❌ **Not supported:**
- CCCL 3.3+ libcu++ features (`cuda::std::*`)
- Limited device lambda support (use functors)

🔧 **Migration steps:**
1. No header path changes needed
2. Replace `cuda::std::*` with `thrust::*` or custom code
3. Use functors instead of device lambdas
4. Test numerical precision (float vs double)

**Effort:** 1-2 hours per sample for basic Thrust, more for CCCL 3.3+ features

**Result:** 90-95% compatibility for standard Thrust patterns. Excellent support!
