<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Troubleshooting

Common errors and fixes for CUDA to HIP conversion.

---

## Common Issue Quick Reference Table

| Error | Likely Cause | Quick Fix | Skill |
|-------|--------------|-----------|-------|
| "operation not supported" at specific line | Deprecated API | Check line number, comment out call | S33 |
| Timeout/hang on atomics sample | System-wide atomics on APU | Use device atomics + fence | S35 |
| Wrong numerical results (large errors) | Wave size hardcoded to 32 | Use warpSize variable | S39 |
| "invalid argument" on cubemap | hipArrayCubemap not supported | Use hipArrayLayered | S36 |
| Different error code than expected | HIP codes differ from CUDA | Accept functional behavior | S34 |
| Memory aperture violation | Placement new on shared mem | Use static shared memory | S37 |
| undeclared identifier 'checkCudaErrors' | Missing hip_helper.h | Add `#include <hip_helper.h>` | S40 |
| GL/gl.h: No such file or directory | Graphics interop | Not portable to Linux ROCm | S41 |
| nvrtc* functions undefined | NVRTC runtime compilation | Port to HIPRTC | S42 |
| npp* functions undefined | NPP library | Use MIVisionX/rocAL | S43 |
| wmma::* or nvcuda::wmma undefined | WMMA/Tensor Cores | Expert rewrite to rocWMMA | S44 |
| cuda/tile header not found | CUDA Tile API | Manual implementation needed | S45 |

---

## Environment Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `hip/hip_runtime.h not found` | Outside ROCm container | Run inside container, check `/opt/rocm/include` |
| `hipcc: not found` | Wrong bash invocation | Use `bash -lc` not `bash -c` |
| Permission denied | UID mismatch | Output to `/tmp` or `--user $(id -u):$(id -g)` |
| `amdgpu-arch` empty | No GPU access | Use compile-only mode (`-c` flag) |
| gfx1151 runtime crash | ROCm too old | Upgrade to ROCm 7.2+ |
| "no kernel image" | Arch mismatch | Rebuild with `--offload-arch=$(amdgpu-arch)` |

---

## Hipify Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `cudaCheck` still in code | Expected behavior | Hipify changes body, not macro name |
| `hipify-clang` fails on include | Missing path | Add `-I/path/to/headers` after `--` |
| `hiphip*` in output | Double translation | Never hipify `.hip` files, use original `.cu` |
| CRLF errors | Windows line endings | Run `dos2unix output.hip` |

---

## Compile Errors

| Error | Fix |
|-------|-----|
| `'__stcs' undeclared` | Replace with `*ptr = val` |
| `'nvcuda::wmma::*' undeclared` | Full rocWMMA rewrite (S25) |
| `'CUBLASLT_*' undeclared` | Map to `HIPBLASLT_*` equivalents |
| `'_Float16' not in scope` | Compile with hipcc (clang), not gcc |
| `'--use_fast_math' unsupported` | Use `-ffast-math` |
| `make_float4: too few args` | Use `make_float4(0,0,0,0)` |
| `'uint' undeclared` | Add `typedef unsigned int uint;` |
| `cooperative_groups.h not found` | Use `hip/hip_cooperative_groups.h` |
| `helper_math.h operator ambiguity` | Use explicit scalar operations |
| Duplicate definitions | Remove dups when merging multi-file projects |
| `'assert' undeclared` | Add `#include <cassert>` |
| `'EXIT_WAIVED' undeclared` | Add `#define EXIT_WAIVED 2` |
| `undefined: main` | Add stub main() for kernel-only files |

---

## Link Errors

| Error | Fix |
|-------|-----|
| `undefined: hipblasLtMatmul` | Add `-lhipblaslt` |
| `undefined: hipblasSgemm` | Add `-lhipblas` |
| `undefined: hipfftPlan1d` | Add `-lhipfft` |
| `undefined: hiprandGenerate` | Add `-lhiprand` |
| `undefined: MIOpen*` | Add `-lMIOpen` |

---

## Runtime Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Garbage output | Warp size assumption | Use `warpSize` built-in, not 32 |
| Wrong shuffle results | 64-bit mask on CDNA | Use `__activemask()` or `~0ULL` |
| `hipErrorNoDevice` | No GPU in container | Expected for compile-only |
| Segfault on launch | Shared memory size | Use `hipFuncSetAttribute` for >48KB |
| Wrong results | Block size | `dim3(128)` = 4 warps NVIDIA, 2 wavefronts AMD |

---

## API Signature Fixes

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaStreamWaitEvent(s, e)` | `hipStreamWaitEvent(s, e, 0)` | Add 3rd param |
| `cudaEventCreate(&e, f)` | `hipEventCreateWithFlags(&e, f)` | Different name |
| `cudaMemAdvise(p, s, a, loc)` | `hipMemAdvise(p, s, a, deviceId)` | int not struct |
| `cudaMalloc3DArray(&a, &d, e)` | `hipMalloc3DArray(&a, &d, e, 0)` | Add 4th param |
| `cudaFuncGetAttributes(&a, k)` | `hipFuncGetAttributes(&a, (void*)k)` | Cast kernel |

---

## cooperative_groups/reduce.h

No HIP equivalent. Manual implementation:

```cpp
// CUDA
#include <cooperative_groups/reduce.h>
float result = cg::reduce(tile, val, cg::plus<float>());

// HIP - manual warp reduction
float sum = val;
for (int i = warpSize/2; i > 0; i /= 2)
    sum += __shfl_down(sum, i);
```

---

## SDK Helper Errors

| Error | Fix |
|-------|-----|
| `checkCudaErrors` undefined | Include `hip_helper.h` |
| `findCudaDevice` undefined | Include `hip_helper.h` |
| `sdkCreateTimer` undefined | Include `hip_helper.h` |
| `sdkLoadPGM` undefined | Include `hip_helper.h` |

See `api-rewrites.md` for complete `hip_helper.h` source.

---

## Quick Fix Commands

```bash
# Add missing typedef
sed -i '1i typedef unsigned int uint;' file.hip

# Fix helper_cuda.h
sed -i 's|#include.*helper_cuda.h.*|#include "hip_helper.h"|g' file.hip

# Fix cooperative_groups header
sed -i 's|<cooperative_groups.h>|<hip/hip_cooperative_groups.h>|g' file.hip

# Fix vector constructors
sed -i 's/make_float4(0)/make_float4(0.0f, 0.0f, 0.0f, 0.0f)/g' file.hip

# Add EXIT_WAIVED
sed -i '1i #define EXIT_WAIVED 2' file.hip

# Remove NVTX calls
sed -i 's/nvtxRangePushA(.*);//' file.hip
sed -i 's/nvtxRangePop();//' file.hip
```

---

---

## Testing Strategies

### Timeout-Based Testing for Hang Detection

Use `timeout` command to detect infinite loops or deadlocks:

```bash
# Test with 30-second timeout
timeout 30 ./sample_hip

# Check exit code
if [ $? -eq 124 ]; then
    echo "TIMEOUT: Sample hung (likely atomics or synchronization issue)"
else
    echo "Completed normally"
fi

# For multiple samples
for sample in ./build/*.hip.out; do
    echo "Testing: $sample"
    timeout 30 "$sample" && echo "PASS" || echo "FAIL/TIMEOUT"
done
```

**Common hang causes:**
- System-wide atomics on APU (S35)
- Missing `__threadfence_system()` after atomics
- Infinite CAS loops without retry limits
- Deadlocks in cooperative group synchronization

### Unbuffered Output for Diagnosis

Use `stdbuf` to see output immediately before a hang:

```bash
# Unbuffered stdout and stderr
stdbuf -oL -eL ./sample_hip

# Shows exactly where the program hangs
```

### Exit Code Patterns

Accept expected errors (some samples test error handling):

```bash
# Run sample and accept specific exit codes
./sample_hip
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 2 ]; then
    echo "PASS (EXIT_WAIVED)"
else
    echo "FAIL: unexpected exit code $EXIT_CODE"
fi
```

### APU Detection

Detect APU vs discrete GPU to apply appropriate fixes:

```bash
# Check if integrated GPU (APU)
rocminfo | grep -A 10 "Agent.*gfx" | grep "Memory.*shared"

# If output contains "shared", it's an APU
# Apply S35 atomics fixes automatically
```

---

## ROCm 7.2 Specific Notes

### Version Requirements

| GPU | Min ROCm | Critical Notes |
|-----|----------|----------------|
| gfx1151 (Strix Halo) | **7.2** | ROCm 6.4 compiles but crashes at runtime |
| gfx1100 (RX 7900) | 5.7 | Works on older versions |
| gfx90a (MI200) | 5.0 | Stable across versions |
| gfx942 (MI300X) | 6.0 | Use 7.2 for latest features |

**CRITICAL:** ROCm < 7.2 for gfx1151 will compile successfully but crash at runtime with cryptic errors.

### Deprecated APIs in ROCm 7.2

| API | Status | Replacement |
|-----|--------|-------------|
| `hipProfilerStart()` | Deprecated (returns hipErrorNotSupported) | roctracer, rocTX |
| `hipProfilerStop()` | Deprecated (returns hipErrorNotSupported) | roctracer, rocTX |

**Detection:** If build succeeds but sample fails with "operation not supported" at specific line, check if it's a deprecated profiler call.

### Recommended Docker Image

```bash
# For gfx1151 and latest features
docker pull rocm/dev-ubuntu-24.04:7.2-complete  # 15GB, ROCm 7.2

# For other GPUs
docker pull rocm/rocm-terminal:latest  # 5GB, ROCm 6.x
```

---

## Debug Checklist

1. Check GPU detected: `amdgpu-arch`
2. Check ROCm version: `hipcc --version` (need 7.2+ for gfx1151)
3. Check binary arch matches: `strings binary | grep gfx`
4. Check warnings file from hipify
5. Verify all `#include` paths exist
6. Check for hardcoded warp size (32)
7. **Test with timeout** to detect hangs (30-60 seconds)
8. **Use unbuffered output** (`stdbuf -oL -eL`) to diagnose hang location
9. **Check for APU** if atomics samples hang (rocminfo)
10. **Verify ROCm 7.2+** if gfx1151 crashes at runtime

---

## Isaac ROS / NITROS Migration

For porting NVIDIA Isaac ROS packages from GXF/NITROS to HIP:

### NITROS API Changes

| Old NITROS (CUDA) | New Pattern (HIP) |
|-------------------|-------------------|
| `.tensors` | `.getTensors()` |
| `.shape.dims` | `.getShape()` |
| `.strides` | `.getStrides()` |
| `cudaStream_t` from NITROS | `hipStream_t` (cast or extract) |

### Decoder Architecture Patterns

**Type A (Complex - GXF-based):**
- Examples: DOPE, DetectNet, UNet
- Requires complete node rewrite
- Extract decoder logic from GXF extensions
- CUDA → HIP conversion
- Custom build configuration

**Type B (Simple - Pure ROS2):**
- Examples: YOLOv8, RT-DETR
- Minimal API updates
- Standard CUDA → HIP changes
- Regular colcon build

### Stream Handling

```cpp
// CUDA (NITROS)
auto stream = msg.stream();
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);

// HIP
hipStream_t hip_stream;  // Extract or create
hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, hip_stream);
```

### ROS2/Colcon Build Pattern

```bash
# Build with HIP support
colcon build \
    --cmake-args \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_HIP_ARCHITECTURES=gfx1151
```

---

## External Library Porting Order

When porting complex projects with dependencies:

```
1. Base libraries (no GPU deps)
   ├── raft (GPU primitives)
   ├── stdgpu (GPU containers)
   └── OpenCV (CPU/GPU hybrid)
       │
2. Mid-level libraries
   ├── cuML → depends on raft
   ├── nvblox → depends on stdgpu
   └── custom decoders
       │
3. Application layer
   ├── Isaac ROS packages
   └── Custom ROS2 nodes
```

**Key Insight:** Port dependencies BEFORE the main library. A single missing dependency (like RAFT) blocks entire project builds.
