<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Hipify Workflow

Automated CUDA to HIP source translation.

---

## Tools

| Tool | Use Case | Speed |
|------|----------|-------|
| `hipify-perl` | Single files, no compile needed | Fast |
| `hipify-clang` | Projects with complex headers | Slower |

Both are in `/opt/rocm/bin/` inside ROCm containers.

---

## hipify-perl (Preferred for Single Files)

```bash
# Basic conversion
hipify-perl input.cu > output.hip

# With stats
hipify-perl --print-stats input.cu > output.hip 2> stats.txt

# Save warnings (your punch list)
hipify-perl input.cu > output.hip 2> warnings.txt
```

---

## Batch Conversion (Sequential - Small Projects)

For small projects (< 10 files):

```bash
for f in *.cu; do
    hipify-perl "$f" > "${f%.cu}.hip" 2> "${f%.cu}_warnings.txt"
done
```

---

## Batch Conversion (Parallel - Large Projects)

For large projects (50+ files), **use parallel processing**:

```bash
# Method 1: xargs with parallel processing (built-in, works everywhere)
find . -name "*.cu" -print0 | \
    xargs -0 -P $(nproc) -I {} bash -c '
        OUT="{}.hip"
        ERR="{}.warnings.txt"
        hipify-perl "{}" > "$OUT" 2> "$ERR"
    '

# Method 2: GNU parallel (if available - fastest)
find . -name "*.cu" | \
    parallel -j $(nproc) \
    'hipify-perl {} > {.}.hip 2> {.}.warnings.txt'

# Method 3: Maximum throughput (use 90% of cores)
CORES=$(nproc)
JOBS=$((CORES * 9 / 10))
find . -name "*.cu" -print0 | \
    xargs -0 -P $JOBS -I {} bash -c '
        hipify-perl "{}" > "{}.hip" 2> "{}.warnings.txt"
    '

# Example speedup: 224 cores → 200+ files/minute vs 10 files/minute sequential
```

---

## Aggregate Warning Analysis

After parallel conversion, analyze all warnings together:

```bash
# Collect all warnings
find . -name "*.warnings.txt" -exec cat {} \; > all_warnings.txt

# Count by issue type
grep -h "warning:" all_warnings.txt | sort | uniq -c | sort -rn

# Find files with specific issues
grep -l "unsupported.*wmma" **/*.warnings.txt
grep -l "unsupported.*__stcs" **/*.warnings.txt

# Summary report
echo "Total files converted: $(find . -name '*.hip' | wc -l)"
echo "Files with warnings: $(find . -name '*.warnings.txt' -type f ! -size 0 | wc -l)"
echo "Top 10 warnings:"
grep -h "warning:" all_warnings.txt | sort | uniq -c | sort -rn | head -10
```

---

## hipify-clang (For Projects)

```bash
# Needs full include paths
hipify-clang input.cu -o output.hip -- \
    -I./include \
    -I/usr/local/cuda/include \
    -std=c++17

# With compile_commands.json
hipify-clang -p build_dir input.cu -o output.hip
```

---

## What Hipify Translates Well

- `cuda*` runtime API → `hip*`
- `cuda*_t` types → `hip*_t`
- Headers: `<cuda_runtime.h>` → `<hip/hip_runtime.h>`
- Error constants: `cudaSuccess` → `hipSuccess`
- Memory directions: `cudaMemcpyHostToDevice` → `hipMemcpyHostToDevice`
- `__shfl_*_sync` warp intrinsics
- cuBLAS handles and basic calls

---

## What Hipify Gets Wrong (Silently)

| Issue | Description |
|-------|-------------|
| User macros | `cudaCheck(hipMalloc(...))` - body translated, name kept |
| `cudaError_t` in templates | Sometimes missed |
| `cudaMemcpyToSymbol` | Semantics differ in older ROCm |
| `cudaStreamPerThread` | Not a first-class HIP concept |

---

## What Hipify Cannot Translate (Warns)

| Warning | Action |
|---------|--------|
| `unsupported: __stcs/__ldcs` | Replace with `*ptr = val` |
| `unsupported: __nanosleep` | Remove |
| `unsupported: __float2bfloat16_rn` | Use `__float2bfloat16` |
| `CUBLASLT_MATMUL_DESC_*` | Manual hipBLASLt rewrite |
| `nvcuda::wmma::*` (no warning) | Full rocWMMA rewrite |
| `asm("...")` PTX | Manual rewrite |

---

## Always Save Warnings

```bash
hipify-perl input.cu > output.hip 2> warnings.txt
```

The stderr file is your punch list for manual fixes. Without it, you don't know what needs attention.

---

## Reading Stats File

```
[HIPIFY] info: file 'input.cu' statistics:
  CONVERTED refs count: 23
  UNCONVERTED refs count: 0
  CONVERSION %: 100.0
```

100% means clean mechanical translation - but doesn't guarantee clean compile.

---

## Avoid Double Translation

Never run hipify on `.hip` files. Always restart from original `.cu`.

```bash
# WRONG
hipify-perl output.hip > output2.hip  # Creates "hiphip*" garbage

# CORRECT
hipify-perl original.cu > output.hip
```

---

## Post-Hipify Fixes

After hipify, common manual fixes needed:

```bash
# Replace helper_cuda.h
sed -i 's|#include.*helper_cuda.h.*|#include "hip_helper.h"|g' output.hip

# Fix cooperative_groups header
sed -i 's|<cooperative_groups.h>|<hip/hip_cooperative_groups.h>|g' output.hip

# Add missing typedef
sed -i '1i typedef unsigned int uint;' output.hip

# Fix vector constructors
sed -i 's/make_float4(0)/make_float4(0.0f, 0.0f, 0.0f, 0.0f)/g' output.hip
```

---

## Complete Conversion Script

```bash
#!/bin/bash
convert_cuda() {
    local src="$1"
    local hip="${src%.cu}.hip"
    
    echo "Converting $src..."
    hipify-perl "$src" > "$hip" 2> "${src%.cu}_warnings.txt"
    
    # Post-process
    sed -i 's|#include.*helper_cuda.h.*|#include "hip_helper.h"|g' "$hip"
    sed -i 's|<cooperative_groups.h>|<hip/hip_cooperative_groups.h>|g' "$hip"
    
    # Check warnings
    if [ -s "${src%.cu}_warnings.txt" ]; then
        echo "  Warnings saved to ${src%.cu}_warnings.txt"
    fi
    
    echo "  Created $hip"
}

# Usage
convert_cuda input.cu
```
