#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Automated issue detection for HIP ports
# Scans converted .hip files for common porting issues (S33-S45)

echo "=== HIP Porting Issue Detection ==="
echo "Scanning for S33-S45 issues..."
echo ""

# S33: Deprecated Profiler APIs
echo "[S33] Deprecated Profiler APIs (ROCm 7.2+):"
PROFILER_HITS=$(grep -rn "hipProfilerStart\|hipProfilerStop\|cudaProfilerStart\|cudaProfilerStop" . --include="*.hip" --include="*.cu" 2>/dev/null)
if [ -n "$PROFILER_HITS" ]; then
    echo "$PROFILER_HITS"
    echo "FIX: Comment out these calls - deprecated in ROCm 7.2"
else
    echo "  None found ✓"
fi
echo ""

# S35: System-Wide Atomics (APU hang risk)
echo "[S35] System-Wide Atomics (APU hang risk):"
ATOMICS_HITS=$(grep -rn "atomic.*_system\s*(" . --include="*.hip" --include="*.cu" 2>/dev/null)
if [ -n "$ATOMICS_HITS" ]; then
    echo "$ATOMICS_HITS"
    echo "FIX: Replace atomicXXX_system() with atomicXXX() + __threadfence_system()"
else
    echo "  None found ✓"
fi
echo ""

# S36: Cubemap Textures
echo "[S36] Cubemap Textures (not supported):"
CUBEMAP_HITS=$(grep -rn "ArrayCubemap\|hipArrayCubemap\|cudaArrayCubemap" . --include="*.hip" --include="*.cu" 2>/dev/null)
if [ -n "$CUBEMAP_HITS" ]; then
    echo "$CUBEMAP_HITS"
    echo "FIX: Convert to hipArrayLayered with 6 layers"
else
    echo "  None found ✓"
fi
echo ""

# S37: Placement New (device-side)
echo "[S37] Device Placement New (memory aperture violations):"
PLACEMENT_NEW=$(grep -rn "new\s*(" . --include="*.hip" --include="*.cu" 2>/dev/null | grep -v "// " | head -10)
if [ -n "$PLACEMENT_NEW" ]; then
    echo "$PLACEMENT_NEW"
    echo "FIX: Use static shared memory instead"
else
    echo "  None found ✓"
fi
echo ""

# S39: Hardcoded Wave Size (32 instead of warpSize)
echo "[S39] Hardcoded Wave Size (causes 27x numerical errors):"
WAVE_SIZE_HITS=$(grep -rn "for.*offset.*=\s*32\s*/\|__shfl.*,\s*32\)" . --include="*.hip" --include="*.cu" 2>/dev/null | head -20)
if [ -n "$WAVE_SIZE_HITS" ]; then
    echo "$WAVE_SIZE_HITS"
    echo "FIX: Replace hardcoded 32 with warpSize"
else
    echo "  None found ✓"
fi
echo ""

# S40: Missing hip_helper.h (51% of build failures)
echo "[S40] Missing hip_helper.h include (51% of build failures):"
MISSING_HELPER=$(grep -rL "hip_helper.h" . --include="*.hip" 2>/dev/null | head -10)
if [ -n "$MISSING_HELPER" ]; then
    echo "Files without hip_helper.h:"
    echo "$MISSING_HELPER"
    echo "FIX: Add #include <hip_helper.h> after other includes"
else
    echo "  All files have hip_helper.h ✓"
fi
echo ""

# S41: Graphics Interop (NOT PORTABLE to Linux ROCm)
echo "[S41] Graphics Interop (NOT PORTABLE to Linux ROCm):"
GRAPHICS_HITS=$(grep -rn "#include\s*<GL/\|#include\s*<EGL/\|#include\s*.*[Vv]ulkan\|#include\s*<d3d" . --include="*.hip" --include="*.cu" 2>/dev/null | head -10)
if [ -n "$GRAPHICS_HITS" ]; then
    echo "$GRAPHICS_HITS"
    echo "WARNING: Graphics interop (17 samples, 8%) cannot be ported to Linux ROCm"
else
    echo "  None found ✓"
fi
echo ""

# S42: NVRTC Runtime Compilation
echo "[S42] NVRTC Runtime Compilation:"
NVRTC_HITS=$(grep -rn "#include\s*<nvrtc\|nvrtcCompile\|nvrtcCreate" . --include="*.hip" --include="*.cu" 2>/dev/null)
if [ -n "$NVRTC_HITS" ]; then
    echo "$NVRTC_HITS"
    echo "FIX: Port to HIPRTC (nvrtcXXX → hiprtcXXX, 2-4 hours effort)"
else
    echo "  None found ✓"
fi
echo ""

# S43: NPP Library Usage
echo "[S43] NPP Library Usage (NVIDIA Performance Primitives):"
NPP_HITS=$(grep -rn "npp[A-Z]\|#include.*npp\.h\|#include.*nppi" . --include="*.hip" --include="*.cu" 2>/dev/null | grep -v "cpp/" | head -10)
if [ -n "$NPP_HITS" ]; then
    echo "$NPP_HITS"
    echo "FIX: Use MIVisionX, rocAL, or OpenCV with HIP (4-8 hours effort)"
else
    echo "  None found ✓"
fi
echo ""

# S44: WMMA/Tensor Cores (EXPERT REWRITE)
echo "[S44] WMMA/Tensor Cores (EXPERT REWRITE - 1-2 weeks):"
WMMA_HITS=$(grep -rn "wmma::\|nvcuda::wmma\|#include.*mma\.h" . --include="*.hip" --include="*.cu" 2>/dev/null)
if [ -n "$WMMA_HITS" ]; then
    echo "$WMMA_HITS"
    echo "WARNING: Expert rewrite required - port to rocWMMA (5 samples, 2-3%)"
else
    echo "  None found ✓"
fi
echo ""

# S45: CUDA Tile API (NO HIP EQUIVALENT)
echo "[S45] CUDA Tile API (NO HIP EQUIVALENT):"
TILE_HITS=$(grep -rn "#include.*cuda/tile\|cuda::cooperative_groups::tile" . --include="*.hip" --include="*.cu" 2>/dev/null)
if [ -n "$TILE_HITS" ]; then
    echo "$TILE_HITS"
    echo "FIX: Manual implementation using warp shuffles (8 samples, 4-5%, 4-8 hours)"
else
    echo "  None found ✓"
fi
echo ""

# Summary
echo "=== Detection Summary ==="
echo "This script scans for 13 common porting issues (S33-S45)"
echo "Run this in your CUDA project directory BEFORE starting conversion"
echo ""
echo "Priority fixes:"
echo "  1. S40: Add hip_helper.h (fixes 51% of build failures)"
echo "  2. S39: Fix wave size (prevents 27x numerical errors)"
echo "  3. S35: Fix APU atomics (prevents infinite hangs)"
echo "  4. S33: Comment deprecated profiler APIs"
echo "  5. S41-S45: Identify non-portable features early"
echo ""
echo "See skill/cuda-hip-porting/SKILL.md for detailed fix patterns"
