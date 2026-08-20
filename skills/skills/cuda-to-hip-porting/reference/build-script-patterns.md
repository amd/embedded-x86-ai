<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Build Script Patterns

Proven patterns from build_hip_samples_v2.sh that delivered 48% improvement (v1→v2).

---

## Exclusion Patterns

**Skip non-portable code early** to avoid wasted build attempts:

```bash
# Patterns that reliably indicate non-portable code
EXCLUDE_PATTERNS="nvrtc|_kernel\.hip|cuda_tile|GLES|EGL|cudla|TensorCore|Vulkan|simpleGL"

# Use with find
find cpp -name "*.hip" | grep -Ev "$EXCLUDE_PATTERNS"

# Example usage
for hipfile in $(find cpp -name "*.hip" | grep -Ev "$EXCLUDE_PATTERNS"); do
    echo "Building: $hipfile"
    hipcc -I. --offload-arch=$GPU "$hipfile" -o "${hipfile%.hip}.out"
done
```

### Pattern Breakdown

| Pattern | Reason | Count | Success Rate |
|---------|--------|-------|--------------|
| `nvrtc` | NVRTC requires HIPRTC rewrite | 9 samples | 0% (4-5%) |
| `_kernel\.hip` | Kernel-only files (no main) | 40+ files | 0% standalone |
| `cuda_tile` | No HIP equivalent | 8 samples | 0%* (*manual impl) |
| `GLES\|EGL` | Graphics not on Linux ROCm | 17 samples | 0% (8.5%) |
| `cudla` | Platform-specific (Tegra) | 10 samples | 0% (5%) |
| `TensorCore` | Requires rocWMMA rewrite | 5 samples | 0% (2.5%) |
| `Vulkan` | Graphics interop | Subset of 17 | 0% |
| `simpleGL` | Specific GL samples | Subset of 17 | 0% |

**Impact:** Saves 61 wasted build attempts (30% of 201 samples)

---

## Multi-File Project Detection

**Problem:** Single .hip files often fail standalone but work when linked properly.

**Solution:** Detect all source files in same directory and compile together.

```bash
# Find all source files in sample directory
sample_dir=$(dirname "$hipfile")
all_sources=$(find "$sample_dir" -maxdepth 1 \( -name "*.hip" -o -name "*.cpp" \) ! -name "*_kernel.hip")

# Compile all together
if [ $(echo "$all_sources" | wc -w) -gt 1 ]; then
    echo "Multi-file project detected: $(echo "$all_sources" | wc -w) files"
    hipcc -I. --offload-arch=$GPU $all_sources -o "${sample_dir}/sample.out"
fi
```

### Kernel File Exclusion

**Critical:** `*_kernel.hip` files should NOT be compiled standalone.

```bash
# WRONG - tries to compile kernel files standalone
find . -name "*.hip" -exec hipcc -o {}.out {} \;
# Error: undefined reference to 'main'

# CORRECT - exclude kernel files
find . -name "*.hip" ! -name "*_kernel.hip" -exec hipcc -o {}.out {} \;
```

**Example:** `BlackScholes` sample has:
- `BlackScholes.hip` (main program)
- `BlackScholes_kernel.hip` (GPU kernels)

Must compile together: `hipcc BlackScholes.hip BlackScholes_kernel.hip -o BlackScholes.out`

---

## Automatic Library Detection

**Problem:** Samples use various libraries - must link correct flags.

**Solution:** Grep source content, auto-add link flags.

```bash
# Function to detect libraries
detect_libraries() {
    local sources="$1"
    local libs=""

    # Check for hipFFT
    if grep -q "hipfft" $sources; then
        libs="$libs -lhipfft"
    fi

    # Check for hipBLAS
    if grep -q "hipblas" $sources; then
        libs="$libs -lhipblas"
    fi

    # Check for hipSPARSE
    if grep -q "hipsparse" $sources; then
        libs="$libs -lhipsparse"
    fi

    # Check for hipRAND
    if grep -q "hiprand" $sources; then
        libs="$libs -lhiprand"
    fi

    # Check for OpenMP
    if grep -q "omp.h" $sources; then
        libs="$libs -fopenmp"
    fi

    # Check for math library
    if grep -q "sin\|cos\|exp\|log" $sources; then
        libs="$libs -lm"
    fi

    echo "$libs"
}

# Usage
LIBS=$(detect_libraries "$all_sources")
hipcc -I. --offload-arch=$GPU $all_sources $LIBS -o sample.out
```

**Impact in v2:** Auto-linked libraries added +5 successful builds

---

## Sample Classification

**Classify samples** by whether they have `main()` to set realistic expectations.

```bash
# Arrays to categorize samples
SAMPLES_WITH_MAIN=()
KERNEL_ONLY=()
EXCLUDED=()

# Scan all samples
for hipfile in $(find cpp -name "*.hip"); do
    # Check exclusion patterns first
    if echo "$hipfile" | grep -Eq "$EXCLUDE_PATTERNS"; then
        EXCLUDED+=("$hipfile")
        continue
    fi

    # Check if has main()
    if grep -q "int main(" "$hipfile" 2>/dev/null; then
        SAMPLES_WITH_MAIN+=("$hipfile")
    else
        KERNEL_ONLY+=("$hipfile")
    fi
done

# Report
echo "=== Sample Classification ==="
echo "With main():    ${#SAMPLES_WITH_MAIN[@]} (expect 40-60% build success)"
echo "Kernel-only:    ${#KERNEL_ONLY[@]} (expect 0% standalone, need multi-file)"
echo "Excluded:       ${#EXCLUDED[@]} (non-portable, don't attempt)"
```

**Expected results:**
- With main(): 10-15 samples (5-10% of total)
- Kernel-only: 80-100 files (need linking)
- Excluded: 61 samples (30% non-portable)

---

## Parallel Processing

**Use all available cores** for I/O-bound tasks like compilation:

```bash
# Check available cores
CORES=$(nproc)
echo "Available cores: $CORES"

# Use 80-90% for parallel builds
PARALLEL_JOBS=$((CORES * 9 / 10))

# Parallel hipify conversion
find . -name "*.cu" -print0 | \
    xargs -0 -P $PARALLEL_JOBS -I {} bash -c '
        hipify-perl "{}" > "{}.hip" 2> "{}.warnings.txt"
    '

# Parallel building (use fewer jobs - more memory intensive)
find . -name "*.hip" ! -name "*_kernel.hip" -print0 | \
    xargs -0 -P $((PARALLEL_JOBS / 2)) -I {} bash -c '
        GPU=$(amdgpu-arch | head -1)
        hipcc -I. --offload-arch=$GPU "{}" -o "{}.out" 2> "{}.build_err.txt"
    '
```

**Performance data:**
- 224 cores → use 180-200 for hipify (I/O bound)
- 224 cores → use 90-112 for compilation (memory bound)
- Speedup: 10-50x for large projects (50+ files)

---

## Docker User Permissions

**Problem:** Root-owned files created inside Docker can't be edited by user.

**Solution:** Always use `--user $(id -u):$(id -g)`.

```bash
# WRONG - creates root-owned files
docker run --rm -v $(pwd):/workspace rocm/dev-ubuntu-24.04:7.2-complete \
    hipcc -o /workspace/test /workspace/test.hip
# Result: test owned by root:root (can't edit/delete)

# CORRECT - files owned by you
docker run --rm --user $(id -u):$(id -g) \
    -v $(pwd):/workspace rocm/dev-ubuntu-24.04:7.2-complete \
    hipcc -o /workspace/test /workspace/test.hip
# Result: test owned by you:your_group (can edit/delete)
```

**Also required:** GPU access

```bash
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete \
    bash -c 'GPU=$(amdgpu-arch | head -1) && hipcc --offload-arch=$GPU test.hip -o test && ./test'
```

---

## Build Success Rate Tracking

**Track success rates** to measure progress:

```bash
# Initialize counters
TOTAL=0
SUCCESS=0
FAIL_LINK=0
FAIL_COMPILE=0
EXCLUDED=0

# Build and track
for hipfile in $(find cpp -name "*.hip"); do
    ((TOTAL++))

    # Check exclusion
    if echo "$hipfile" | grep -Eq "$EXCLUDE_PATTERNS"; then
        ((EXCLUDED++))
        continue
    fi

    # Attempt build
    hipcc -I. --offload-arch=$GPU "$hipfile" -o "${hipfile%.hip}.out" 2>"${hipfile%.hip}.err"

    if [ $? -eq 0 ]; then
        ((SUCCESS++))
    else
        # Classify failure
        if grep -q "undefined reference to 'main'" "${hipfile%.hip}.err"; then
            ((FAIL_LINK++))  # Kernel-only file
        else
            ((FAIL_COMPILE++))  # Real compilation error
        fi
    fi
done

# Report
echo "=== Build Results ==="
echo "Total files:      $TOTAL"
echo "Excluded:         $EXCLUDED (non-portable)"
echo "Built:            $SUCCESS ($(( SUCCESS * 100 / (TOTAL - EXCLUDED) ))%)"
echo "Link errors:      $FAIL_LINK (kernel-only, need multi-file)"
echo "Compile errors:   $FAIL_COMPILE (need fixes)"
```

**Interpretation:**
- Built %: 5-10% = normal, 40-50% with CMake
- Link errors: Expected - kernel-only files
- Compile errors: Apply fixes (S33-S45)

---

## Complete Build Script Template

Based on build_hip_samples_v2.sh:

```bash
#!/bin/bash
# HIP sample builder with automatic detection

# Configuration
EXCLUDE_PATTERNS="nvrtc|_kernel\.hip|cuda_tile|GLES|EGL|cudla|TensorCore|Vulkan|simpleGL"
GPU=$(amdgpu-arch 2>/dev/null | head -1)
CORES=$(nproc)
PARALLEL_JOBS=$((CORES * 9 / 10))

echo "=== HIP Sample Builder ==="
echo "GPU: ${GPU:-No GPU detected (compile-only mode)}"
echo "Cores: $CORES (using $PARALLEL_JOBS parallel jobs)"
echo ""

# Function: Detect libraries
detect_libraries() {
    local sources="$*"
    local libs=""
    grep -q "hipfft" $sources && libs="$libs -lhipfft"
    grep -q "hipblas" $sources && libs="$libs -lhipblas"
    grep -q "hipsparse" $sources && libs="$libs -lhipsparse"
    grep -q "hiprand" $sources && libs="$libs -lhiprand"
    grep -q "omp.h" $sources && libs="$libs -fopenmp"
    echo "$libs"
}

# Build samples
for hipfile in $(find cpp -name "*.hip" | grep -Ev "$EXCLUDE_PATTERNS"); do
    # Skip kernel-only files
    echo "$hipfile" | grep -q "_kernel\.hip" && continue

    # Check for multi-file project
    sample_dir=$(dirname "$hipfile")
    all_sources=$(find "$sample_dir" -maxdepth 1 \( -name "*.hip" -o -name "*.cpp" \) ! -name "*_kernel.hip")

    # Detect libraries
    LIBS=$(detect_libraries $all_sources)

    # Build
    echo "Building: $(basename $sample_dir)"
    if [ -n "$GPU" ]; then
        hipcc -I. --offload-arch=$GPU $all_sources $LIBS -o "${sample_dir}/sample.out" 2>"${sample_dir}/build.err"
    else
        hipcc -I. -c $all_sources $LIBS -o "${sample_dir}/sample.o" 2>"${sample_dir}/build.err"
    fi

    if [ $? -eq 0 ]; then
        echo "  SUCCESS"
    else
        echo "  FAILED (see ${sample_dir}/build.err)"
    fi
done
```

---

## Summary: Key Patterns

1. **Exclusion patterns** - Skip 61 non-portable samples (30%), save time
2. **Multi-file detection** - Link properly, +15-20 builds (40-50% success)
3. **Library auto-detection** - Add correct flags, +5 builds
4. **Kernel file exclusion** - Don't compile `*_kernel.hip` standalone
5. **Parallel processing** - Use 80-90% of cores, 10-50x speedup
6. **Docker permissions** - Always `--user $(id -u):$(id -g)`
7. **Success rate tracking** - Measure progress, identify issues

**Result:** v1 (20%) → v2 (30%) = **+48% improvement**
