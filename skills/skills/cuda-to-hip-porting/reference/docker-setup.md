<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Docker Setup

Complete container setup for CUDA to HIP conversion with auto GPU detection.

---

## Images

| Purpose | Image | Size |
|---------|-------|------|
| CUDA compile | `nvidia/cuda:13.2.1-devel-ubuntu22.04` | ~7GB |
| ROCm general | `rocm/rocm-terminal:latest` | ~5GB |
| **ROCm 7.2+ (gfx1151)** | `rocm/dev-ubuntu-24.04:7.2-complete` | ~15GB |

---

## Pull Image (with check)

```bash
if ! docker images rocm/dev-ubuntu-24.04:7.2-complete -q | grep -q .; then
    echo "Pulling ROCm 7.2 image..."
    docker pull rocm/dev-ubuntu-24.04:7.2-complete
else
    echo "Image already exists"
fi
```

---

## Permission Management

**CRITICAL**: Always use `--user $(id -u):$(id -g)` in Docker commands that create files.

**Why this matters**:
- Docker containers run as root by default
- Files created in mounted volumes become owned by root
- This causes "Permission denied" errors when accessing files outside Docker
- Using `--user` flag matches container UID/GID to your host user

**When to use**:
- ✅ Always use for commands that create/modify files (hipify, hipcc -o)
- ❌ Optional for read-only operations (amdgpu-arch, docker images)

**Example**:
```bash
# WRONG - creates root-owned files
docker run --rm -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete \
    hipify-perl input.cu > output.hip

# CORRECT - files owned by you
docker run --rm --user $(id -u):$(id -g) \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete \
    hipify-perl input.cu > output.hip
```

---

## GPU Detection

```bash
# Detect GPU architecture in Docker
docker run --rm --device=/dev/kfd --device=/dev/dri \
    rocm/dev-ubuntu-24.04:7.2-complete amdgpu-arch
# Returns: gfx1151, gfx90a, gfx1100, etc.

# Alternative methods inside container
rocminfo | grep -m1 "gfx"
rocm-smi --showproductname
```

---

## Auto-Detect, Compile, Run

```bash
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete bash -c '
        GPU=$(amdgpu-arch | head -1)
        if [ -n "$GPU" ]; then
            echo "Detected GPU: $GPU"
            hipcc -I. --offload-arch=$GPU -w -o output output.hip && ./output
        else
            echo "No GPU detected, compile-only mode"
            hipcc -I. -c -w -o output.o output.hip
        fi
    '
```

---

## Smart Recompile (check binary arch)

```bash
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete bash -c '
        GPU=$(amdgpu-arch | head -1)
        BIN="output_hip"
        SRC="output.hip"

        if [ -z "$GPU" ]; then
            echo "No GPU, compile-only"
            hipcc -I. -c -w -o output.o "$SRC"
            exit 0
        fi

        # Check if recompile needed
        BIN_ARCH=$(strings "$BIN" 2>/dev/null | grep -oE "gfx[0-9]+" | head -1)
        if [ ! -f "$BIN" ] || [ "$BIN_ARCH" != "$GPU" ]; then
            echo "Compiling for $GPU (was: ${BIN_ARCH:-none})..."
            hipcc -I. --offload-arch=$GPU -w -o "$BIN" "$SRC"
        else
            echo "Binary already matches $GPU"
        fi
        ./"$BIN"
    '
```

---

## Compile Only (No GPU Required)

```bash
docker run --rm --user $(id -u):$(id -g) \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete \
    hipcc -I. -c -w -o output.o output.hip
```

---

## Multi-Arch Build

```bash
docker run --rm --user $(id -u):$(id -g) \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete bash -c '
        hipcc -I. \
            --offload-arch=gfx90a \
            --offload-arch=gfx1100 \
            --offload-arch=gfx1151 \
            -w -o output output.hip
    '
```

---

## Batch Test All Files

```bash
docker run --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete bash -c '
        GPU=$(amdgpu-arch | head -1)
        [ -z "$GPU" ] && { echo "No GPU"; exit 1; }

        PASS=0; FAIL=0
        for f in *.hip; do
            [ -f "$f" ] || continue
            echo -n "Testing $f... "
            if hipcc -I. --offload-arch=$GPU -w -o /tmp/test "$f" 2>/dev/null; then
                if timeout 30 /tmp/test >/dev/null 2>&1; then
                    echo "PASS"; ((PASS++))
                else
                    echo "RUN_FAIL"; ((FAIL++))
                fi
            else
                echo "COMPILE_FAIL"; ((FAIL++))
            fi
        done
        echo "Results: $PASS passed, $FAIL failed"
    '
```

---

## Interactive Container

```bash
docker run -it --rm --user $(id -u):$(id -g) \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v $(pwd):/workspace \
    -w /workspace \
    rocm/dev-ubuntu-24.04:7.2-complete

# Inside container:
hipcc --version
amdgpu-arch
rocm-smi
```

---

## ROCm Version Compatibility

| GPU | Min ROCm | Notes |
|-----|----------|-------|
| gfx1151 (Strix Halo) | **7.2** | Earlier versions crash at runtime |
| gfx1100 (RX 7900) | 5.7 | |
| gfx90a (MI200) | 5.0 | |
| gfx942 (MI300X) | 6.0 | |

---

## ROCm 6.x vs 7.x ABI Incompatibility

**CRITICAL:** ROCm 7.x has ABI breaks from 6.x. Libraries and pip packages are NOT cross-compatible.

### Symptoms

```bash
# Error when using ROCm 6.x packages with 7.x runtime:
ImportError: libhipblas.so.2: cannot open shared object file
# ROCm 6.x has libhipblas.so.2, ROCm 7.x has libhipblas.so.3

# Or runtime crashes like:
hipErrorNoBinaryForGpu: Unable to find code object for all current devices!
```

### Rules

| Package Source | Works With | Does NOT Work With |
|----------------|------------|-------------------|
| `pip install onnxruntime-rocm` (6.x build) | ROCm 6.x | ROCm 7.x ❌ |
| ROCm 7.x container packages | ROCm 7.x | ROCm 6.x ❌ |
| Source builds in container | Same ROCm version | Different versions ❌ |

### Best Practices

1. **Always check ROCm version before installing packages:**
   ```bash
   cat /opt/rocm/.info/version
   # Example: 7.2.0-39
   ```

2. **Use version-matched pip repositories:**
   ```bash
   # For ROCm 7.x
   pip install onnxruntime-rocm --index-url https://download.pytorch.org/whl/rocm7.0

   # For ROCm 6.x
   pip install onnxruntime-rocm --index-url https://download.pytorch.org/whl/rocm6.2
   ```

3. **Rebuild from source when switching ROCm versions:**
   - Delete old build artifacts
   - Re-run cmake with clean cache
   - Full rebuild required

---

## hipcc Directory Structure Fix

**Problem:** hipcc in some containers expects files in `/opt/rocm/hip/` but the ROCm installation uses versioned paths like `/opt/rocm-7.2.0/`.

### Symptoms

```bash
hipcc: error: cannot find HIP runtime headers
hipcc: error: cannot open '/opt/rocm/hip/include/hip/hip_runtime.h'
```

### Fix with Symlinks

```bash
# Create expected directory structure
mkdir -p /opt/rocm/hip/share/hip

# Link include and lib directories
ln -sf /opt/rocm-7.2.0/include /opt/rocm/hip/include
ln -sf /opt/rocm-7.2.0/lib /opt/rocm/hip/lib

# For share directory (contains cmake configs)
ln -sf /opt/rocm-7.2.0/share/hip/* /opt/rocm/hip/share/hip/
```

### Alternative: Set Environment Variables

```bash
export HIP_PATH=/opt/rocm-7.2.0
export ROCM_PATH=/opt/rocm-7.2.0
export HIP_CLANG_PATH=/opt/rocm-7.2.0/llvm/bin
```

### Dockerfile Pattern

```dockerfile
FROM rocm/dev-ubuntu-24.04:7.2-complete

# Fix hipcc path expectations
RUN mkdir -p /opt/rocm/hip/share/hip && \
    ln -sf /opt/rocm-7.2.0/include /opt/rocm/hip/include && \
    ln -sf /opt/rocm-7.2.0/lib /opt/rocm/hip/lib && \
    ln -sf /opt/rocm-7.2.0/share/hip/* /opt/rocm/hip/share/hip/
```

---

## Common Issues

| Issue | Fix |
|-------|-----|
| `nvcc: not found` | Use `-devel-` tag, not `-runtime-` |
| `hipcc: not found` | Use `bash -lc` not `bash -c` |
| Permission denied | Output to `/tmp` or use `--user $(id -u):$(id -g)` |
| `amdgpu-arch` empty | No GPU access, use compile-only mode |
| gfx1151 runtime crash | Upgrade to ROCm 7.2+ |
| `rocm-terminal` UID issues | Runs as UID 1000, may differ from host |
