<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# ESS Model Conversion: TensorRT Custom Plugins to Standard ONNX

## Overview

Convert NVIDIA ESS stereo depth models containing proprietary TensorRT custom operators to standard ONNX for cross-platform execution on AMD MIGraphX, ONNX Runtime, and OpenVINO.

---

## Problem

```
NGC Model (.onnx) ──► TensorRT + ess_plugins.so ──► NVIDIA Only
                              ╳ MIGraphX (AMD)
                              ╳ ONNX Runtime
                              ╳ OpenVINO

Error: "Unknown operator: FusedEinsum2Softmax"
```

---

## Custom Plugins (5 Instances)

| # | Plugin Name | Instances | Purpose |
|---|-------------|-----------|---------|
| 1 | `FusedConcatConv3x3` | 1 | Multi-scale feature fusion |
| 2 | `FusedEinsum2Softmax` | 1 | Stereo cost volume computation |
| 3 | `FusedConcat2c128c64Conv3x3c64` | 1 | Feature refinement (192→64 ch) |
| 4 | `FusedConcat2c64c64Conv3x3c64` | 2 | Feature refinement (128→64 ch) |
| **Total** | | **5** | |

---

## Standard ONNX Replacement (15 Operators)

| Custom Plugin | Standard ONNX Equivalent | Ops |
|---------------|--------------------------|-----|
| `FusedConcatConv3x3` | Concat → Conv → Conv | 3 |
| `FusedEinsum2Softmax` | Einsum → Transpose → Softmax | 3 |
| `FusedConcat2c128c64Conv3x3c64` | Concat → Conv → Relu | 3 |
| `FusedConcat2c64c64Conv3x3c64` (×2) | Concat → Conv → Relu | 6 |
| **Total** | | **15** |

---

## Decomposition Details

### FusedConcatConv3x3
```
Concat(x1..x5, axis=1) → Conv(3×3, pad=1) → Conv(1×1)
```

### FusedEinsum2Softmax
```
Einsum("bchw,bchd->bhwd") → Transpose([0,3,1,2]) → Softmax(axis=1)
```
- Computes stereo correlation cost volume
- Transpose converts [B,H,W,D] → [B,D,H,W] for NCHW format

### FusedConcat2cXXXConv3x3cYYY
```
Concat(x1, x2, axis=1) → Conv(3×3, pad=1) → Relu
```

---

## Quick Start

```bash
# 1. Decompose custom operators
python tools/decompose_custom_ops.py ess.onnx ess_decomposed.onnx

# 2. Simplify with fixed shapes (required for MIGraphX)
python -c "
from onnxsim import simplify
import onnx
model = onnx.load('ess_decomposed.onnx')
model_simp, _ = simplify(model, overwrite_input_shapes={
    'input_left': [1, 3, 576, 960],
    'input_right': [1, 3, 576, 960]
})
onnx.save(model_simp, 'ess_simplified.onnx')
"

# 3. Test on MIGraphX
python -c "
import migraphx
model = migraphx.parse_onnx('ess_simplified.onnx')
model.compile(migraphx.get_target('gpu'))
print('Success!')
"
```

---

## Model Statistics

| Metric | Original | Converted |
|--------|----------|-----------|
| Custom Operators | 5 | 0 |
| Total Nodes | 295 | 305 |
| Model Size | 66 MB | 66 MB |
| TensorRT Compatible | Yes (with plugins) | Yes |
| MIGraphX Compatible | No | **Yes** |

---

## Performance

### Subsequent Runs

| Model | Orin Original (ms) | Orin Converted (ms) | Strix Converted (ms) |
|-------|-------------------|---------------------|----------------------|
| ESS (576×960) | 8.62 | 10.12 (+17.4%) | 52.95 |
| Light ESS (288×480) | 2.84 | 2.96 (+4.2%) | 13.72 |

### Cold Start / First Inference

| Model | Orin Original (ms) | Orin Converted (ms) | Strix Converted (ms) |
|-------|-------------------|---------------------|----------------------|
| ESS | 8.62 | 10.12 | 1061.82 |
| Light ESS | 2.84 | 2.96 | 671.72 |

### Compile/Build Time

| Model | Orin Original (s) | Orin Converted (s) | Strix Converted (s) |
|-------|-------------------|-------------------|---------------------|
| ESS | 99.7 | 113.3 | 59.0 |
| Light ESS | 58.9 | 62.2 | 26.2 |

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Unknown operator: FusedXXX` | Run tools/decompose_custom_ops.py |
| `Shape inference failed` | Simplify with fixed input shapes |
| `Einsum not supported` | Update opset to 17 |

---

## References

- [NGC dnn_stereo_disparity](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity)
- [Isaac ROS ESS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_stereo_depth)
- [ONNX Operators](https://onnx.ai/onnx/operators/)

---

*Version: 1.0 | Date: 2026-07-03*
