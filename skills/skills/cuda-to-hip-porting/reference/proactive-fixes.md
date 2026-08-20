<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# cuda-to-hip-porting: Proactive Fix Automation (S67–S71)

Full scripts for proactive fixes and large-project parallelization.
Apply **AFTER hipify, BEFORE the first build** to achieve 70–85% first-attempt success.

---

## S67: Wave Size Portability Fixes (MUST DO)

AMD uses 64-thread waves, NVIDIA uses 32-thread warps. Hardcoded 32 causes **27x numerical errors**.

```bash
find . -name "*.hip" -exec sed -i \
    -e 's/threadIdx\.x % 32/threadIdx.x % warpSize/g' \
    -e 's/threadIdx\.x \/ 32/threadIdx.x \/ warpSize/g' {} \;

find . -name "*.hip" -exec sed -i \
    -e 's/__shfl_down(\([^,]*\), \([^,]*\), 32)/__shfl_down(\1, \2, warpSize)/g' \
    -e 's/__shfl_up(\([^,]*\), \([^,]*\), 32)/__shfl_up(\1, \2, warpSize)/g' \
    -e 's/__shfl_xor(\([^,]*\), \([^,]*\), 32)/__shfl_xor(\1, \2, warpSize)/g' {} \;

find . -name "*.hip" -exec sed -i \
    's/for.*offset = 32 \/ 2/for (int offset = warpSize \/ 2/g' {} \;
find . -name "*.hip" -exec sed -i \
    's/for.*offset = 16;/for (int offset = warpSize\/2;/g' {} \;
```

---

## S68: Kernel Header cooperative_groups.h Fix (MUST DO)

```bash
find . -name "*.cuh" -exec grep -l "cooperative_groups\.h" {} \; | while read f; do
    sed -i 's|#include <cooperative_groups.h>|#ifdef __HIP_PLATFORM_AMD__\n#include <hip/hip_cooperative_groups.h>\n#else\n#include <cooperative_groups.h>\n#endif|g' "$f"
done

find . -name "*.h" -exec grep -l "cooperative_groups\.h" {} \; | while read f; do
    sed -i 's|#include <cooperative_groups.h>|#ifdef __HIP_PLATFORM_AMD__\n#include <hip/hip_cooperative_groups.h>\n#else\n#include <cooperative_groups.h>\n#endif|g' "$f"
done
```

---

## S69: Type Compatibility Typedefs (MUST DO)

```bash
find . -name "*.cuh" -exec grep -l "cudaTextureObject_t\|cudaDeviceProp" {} \; | while read f; do
    if ! grep -q "typedef hipTextureObject_t cudaTextureObject_t" "$f"; then
        sed -i '/#include/{ 
            :loop
            n
            /#include/b loop
            i\
// HIP compatibility typedefs\
#ifdef __HIP_PLATFORM_AMD__\
typedef hipTextureObject_t cudaTextureObject_t;\
typedef hipDeviceProp_t cudaDeviceProp;\
#endif
        }' "$f"
    fi
done
```

---

## S70: Deprecated Profiler API Fix (MUST DO for ROCm 7.2+)

```bash
find . -name "*.hip" -exec sed -i \
    -e 's/checkCudaErrors(hipProfilerStart());/\/\/ Note: hipProfilerStart deprecated in ROCm 7.2\n    \/\/ checkCudaErrors(hipProfilerStart());/g' \
    -e 's/checkCudaErrors(hipProfilerStop());/\/\/ checkCudaErrors(hipProfilerStop());/g' \
    -e 's/hipProfilerStart();/\/\/ hipProfilerStart(); \/\/ deprecated ROCm 7.2/g' \
    -e 's/hipProfilerStop();/\/\/ hipProfilerStop(); \/\/ deprecated ROCm 7.2/g' {} \;
```

---

## S71: Multi-File Project Auto-Detection

```bash
echo "=== Kernel-only files (need linking) ==="
for f in $(find . -name "*.hip"); do
    if ! grep -q "int main\|void main" "$f"; then echo "$f"; fi
done

echo "=== Multi-file project directories ==="
find . -type d -exec sh -c '
    hip_count=$(find "$1" -maxdepth 1 -name "*.hip" 2>/dev/null | wc -l)
    cpp_count=$(find "$1" -maxdepth 1 -name "*.cpp" 2>/dev/null | wc -l)
    [ "$hip_count" -gt 0 ] && [ "$cpp_count" -gt 0 ] && echo "$1: $hip_count .hip + $cpp_count .cpp"
' _ {} \;
```

---

## Complete proactive_fixes.sh Script

Save as `proactive_fixes.sh` and run immediately after hipify:

```bash
#!/bin/bash
# proactive_fixes.sh — Run AFTER hipify, BEFORE building
# Usage: ./proactive_fixes.sh /path/to/converted/project

PROJECT="${1:-.}"
cd "$PROJECT" || exit 1
echo "=== Proactive HIP Fixes (S67-S71) ==="

echo "[S67] Fixing hardcoded warp size 32 → warpSize..."
count=0
for f in $(find . -name "*.hip" -o -name "*.cuh"); do
    if grep -q "threadIdx.x % 32\|threadIdx.x / 32\|__shfl.*32)" "$f" 2>/dev/null; then
        sed -i \
            -e 's/threadIdx\.x % 32/threadIdx.x % warpSize/g' \
            -e 's/threadIdx\.x \/ 32/threadIdx.x \/ warpSize/g' \
            -e 's/__shfl_down(\([^,]*\), \([^,]*\), 32)/__shfl_down(\1, \2, warpSize)/g' \
            -e 's/__shfl_up(\([^,]*\), \([^,]*\), 32)/__shfl_up(\1, \2, warpSize)/g' \
            -e 's/__shfl_xor(\([^,]*\), \([^,]*\), 32)/__shfl_xor(\1, \2, warpSize)/g' "$f"
        ((count++))
    fi
done
echo "  Fixed $count files"

echo "[S68] Fixing cooperative_groups.h includes..."
count=0
for f in $(find . \( -name "*.cuh" -o -name "*.h" \) | xargs grep -l "cooperative_groups\.h" 2>/dev/null); do
    if ! grep -q "__HIP_PLATFORM_AMD__" "$f"; then
        sed -i 's|#include <cooperative_groups.h>|#ifdef __HIP_PLATFORM_AMD__\n#include <hip/hip_cooperative_groups.h>\n#else\n#include <cooperative_groups.h>\n#endif|g' "$f"
        ((count++))
    fi
done
echo "  Fixed $count files"

echo "[S69] Adding type compatibility typedefs..."
count=0
for f in $(find . -name "*.cuh" | xargs grep -l "cudaTextureObject_t" 2>/dev/null); do
    if ! grep -q "typedef hipTextureObject_t" "$f"; then
        sed -i '1i\
#ifdef __HIP_PLATFORM_AMD__\
typedef hipTextureObject_t cudaTextureObject_t;\
typedef hipDeviceProp_t cudaDeviceProp;\
#endif' "$f"
        ((count++))
    fi
done
echo "  Fixed $count files"

echo "[S70] Commenting out deprecated profiler APIs..."
count=$(grep -r "hipProfilerStart\|hipProfilerStop" --include="*.hip" . 2>/dev/null | wc -l)
find . -name "*.hip" -exec sed -i \
    -e 's/^\([[:space:]]*\)checkCudaErrors(hipProfilerStart());/\1\/\/ checkCudaErrors(hipProfilerStart()); \/\/ deprecated ROCm 7.2/g' \
    -e 's/^\([[:space:]]*\)checkCudaErrors(hipProfilerStop());/\1\/\/ checkCudaErrors(hipProfilerStop()); \/\/ deprecated ROCm 7.2/g' \
    -e 's/^\([[:space:]]*\)hipProfilerStart();/\1\/\/ hipProfilerStart(); \/\/ deprecated ROCm 7.2/g' \
    -e 's/^\([[:space:]]*\)hipProfilerStop();/\1\/\/ hipProfilerStop(); \/\/ deprecated ROCm 7.2/g' {} \;
echo "  Processed $count occurrences"

echo "[S71] Detecting multi-file projects..."
multifile=0
for dir in $(find . -type d); do
    hip_count=$(find "$dir" -maxdepth 1 -name "*.hip" 2>/dev/null | wc -l)
    cpp_count=$(find "$dir" -maxdepth 1 -name "*.cpp" 2>/dev/null | wc -l)
    if [ "$hip_count" -gt 0 ] && [ "$cpp_count" -gt 0 ]; then
        echo "  Multi-file: $dir ($hip_count .hip + $cpp_count .cpp)"
        ((multifile++))
    fi
done
echo "  Found $multifile multi-file project directories"

echo "=== Proactive fixes complete ==="
echo "Now run: hipcc -I./Common --offload-arch=\$GPU -w -o sample sample.hip"
```

---

## Large Project Conversion (50+ files)

Parallel processing provides 10–50× speedup over sequential conversion.

```bash
# Check available cores
nproc

# Parallel hipify (use 90% of cores)
CORES=$(nproc)
PARALLEL_JOBS=$((CORES * 9 / 10))
find . -name "*.cu" -print0 | \
    xargs -0 -P $PARALLEL_JOBS -I {} bash -c '
        hipify-perl "{}" > "{}.hip" 2> "{}.warnings.txt"
    '

# Aggregate warnings for bulk analysis
cat **/*.warnings.txt | grep "warning:" | sort | uniq -c | sort -rn

# Parallel build (CMake — better success rate)
cmake --build . --parallel $(nproc)
```

---

## Efficiency Rules

| DO | DON'T |
|----|-------|
| `find . -name "*.cu" \| xargs -P $(nproc) -I {} hipify-perl {}` | Sequential `for file in *.cu` loops |
| Collect all warnings once with `cat **/*.warnings.txt` | Analyze per-file individually |
| Output to `.build/` in project dir | Output to `/tmp/` (collides on shared systems) |
