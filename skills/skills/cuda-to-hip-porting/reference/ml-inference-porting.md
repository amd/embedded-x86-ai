<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# ML Inference Framework Porting (S46)

Porting machine learning inference frameworks from CUDA to ROCm, focusing on ONNX Runtime and MIGraphX.

---

## ONNX Runtime: ROCm EP to MIGraphX EP

### Background

- **ROCm EP (Execution Provider)**: Deprecated in ONNX Runtime 1.18+
- **MIGraphX EP**: Recommended replacement for AMD GPU inference
- **CPU EP**: Always available as fallback

### Installation

```bash
# ROCm 7.x (check your version first)
cat /opt/rocm/.info/version

# Install from ROCm-matched repository
pip install onnxruntime-migraphx \
    --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/

# Or build from source for custom ROCm versions
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --use_migraphx --migraphx_home /opt/rocm
```

### Session Setup

```python
import onnxruntime as ort

# BEFORE - CUDA or deprecated ROCm EP
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider']  # NVIDIA
    # or ['ROCMExecutionProvider']        # Deprecated AMD
)

# AFTER - MIGraphX EP (recommended for AMD)
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ('MIGraphXExecutionProvider', {
            'device_id': 0,
            'fp16_enable': True,           # Enable FP16 for 2x speedup
            'int8_enable': False,          # INT8 quantization (if model supports)
        }),
        'CPUExecutionProvider'             # Fallback for unsupported ops
    ]
)

# Check which provider is active
print(session.get_providers())  # Should show MIGraphXExecutionProvider
```

---

## Performance Tuning

### Warmup Strategy

**Critical:** First inference includes graph compilation (10-60 seconds). Run warmup before latency-sensitive operations.

```python
import numpy as np

# Create session
session = ort.InferenceSession("model.onnx", providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider'])

# Get input shape from model
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
# Handle dynamic dims (replace None/-1 with typical values)
input_shape = [1 if d is None or d < 0 else d for d in input_shape]

# Warmup run (triggers MIGraphX compilation)
dummy_input = np.random.randn(*input_shape).astype(np.float32)
_ = session.run(None, {input_name: dummy_input})

print("Warmup complete - ready for inference")
```

### FP16 for 2x Throughput

```python
# Enable FP16 mixed precision
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ('MIGraphXExecutionProvider', {
            'fp16_enable': True,
        }),
        'CPUExecutionProvider'
    ]
)
```

**Notes:**
- FP16 requires GPU with native FP16 support (all recent AMD GPUs)
- May cause minor accuracy differences (~0.1%)
- Check model outputs after enabling

### Session Reuse

**Do:** Reuse sessions across inferences
**Don't:** Create new sessions per inference (triggers recompilation)

```python
# GOOD - create once, reuse
class InferenceEngine:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider']
        )
        self._warmup()

    def _warmup(self):
        # Run one inference to compile graph
        dummy = self._create_dummy_input()
        self.session.run(None, dummy)

    def infer(self, input_data):
        return self.session.run(None, input_data)

# Create once at startup
engine = InferenceEngine("model.onnx")

# Reuse for all inferences
for data in batch:
    result = engine.infer(data)
```

---

## MIGraphX Limitations

### Graph Complexity Limits

**Problem:** Models with >4000 operators may fall back to CPU.

| Model Size | MIGraphX Behavior | Workaround |
|------------|-------------------|------------|
| <1000 ops | Full GPU execution | None needed |
| 1000-4000 ops | Usually GPU | Monitor perf |
| >4000 ops | May fall back to CPU | Split model or simplify |

**Detection:**

```python
# Check for CPU fallback
providers = session.get_providers()
if 'CPUExecutionProvider' in providers and 'MIGraphXExecutionProvider' not in providers:
    print("WARNING: Model fell back to CPU - may be too complex")

# Or check MIGraphX logs
import os
os.environ['MIGRAPHX_TRACE_COMPILE'] = '1'
# Then examine output during session creation
```

### Unsupported Operators

Some ONNX operators lack MIGraphX kernels:

| Operator | Status | Workaround |
|----------|--------|------------|
| `NonMaxSuppression` | Partial | CPU fallback |
| `TopK` (large K) | Limited | Reduce K or CPU |
| `Einsum` (complex) | Partial | Expand to matmul |
| `DynamicQuantizeLinear` | No | Use static quant |

**Check operator support:**

```bash
# List MIGraphX supported ops
migraphx-driver perf --onnx model.onnx --list-ops
```

---

## Provider Selection Guide

### Decision Tree

```
Is model <4000 operators?
├── Yes → Use MIGraphXExecutionProvider
│   ├── Need FP16? → Enable fp16_enable
│   └── Need INT8? → Enable int8_enable (requires quantized model)
└── No → Consider:
    ├── Split model into sub-graphs
    ├── Use CPUExecutionProvider for complex sections
    └── Simplify model (ONNX optimizer)
```

### Configuration Template

```python
def get_optimal_providers(model_complexity, needs_speed=True):
    """Select providers based on model and requirements."""

    providers = []

    if model_complexity < 4000:  # ops count
        migraphx_options = {
            'device_id': 0,
            'fp16_enable': needs_speed,
        }
        providers.append(('MIGraphXExecutionProvider', migraphx_options))

    # Always include CPU fallback
    providers.append('CPUExecutionProvider')

    return providers

# Usage
providers = get_optimal_providers(model_complexity=2500, needs_speed=True)
session = ort.InferenceSession("model.onnx", providers=providers)
```

---

## Debugging

### Enable Verbose Logging

```python
import onnxruntime as ort

# Set logging level
ort.set_default_logger_severity(0)  # 0=Verbose, 1=Info, 2=Warning, 3=Error

# Or via environment
import os
os.environ['ORT_LOG_LEVEL'] = 'VERBOSE'
```

### Profile Inference

```python
# Enable profiling
options = ort.SessionOptions()
options.enable_profiling = True

session = ort.InferenceSession(
    "model.onnx",
    sess_options=options,
    providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider']
)

# Run inference
result = session.run(None, input_data)

# Get profiling data
profile_file = session.end_profiling()
print(f"Profile saved to: {profile_file}")
```

### Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `MIGraphX provider not found` | Not installed | `pip install onnxruntime-migraphx` |
| `hipErrorNoBinaryForGpu` | ROCm version mismatch | Rebuild for your ROCm version |
| `Graph compilation failed` | Unsupported op | Check ops with `--list-ops` |
| `Out of memory` | Model too large | Enable FP16, reduce batch |

---

## PyTorch to ONNX for MIGraphX

### Export Best Practices

```python
import torch

model = MyModel()
model.eval()

# Dummy input matching expected inference shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export with MIGraphX-friendly settings
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,              # Use latest supported opset
    do_constant_folding=True,      # Optimize constants
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={                 # If batch size varies
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)
```

### Optimize ONNX for MIGraphX

```python
import onnx
from onnxruntime.transformers import optimizer

# Load model
model = onnx.load("model.onnx")

# Optimize for inference
optimized = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',  # or 'gpt2', 'vit', etc.
    opt_level=2
)
optimized.save_model_to_file("model_optimized.onnx")
```

---

## MIGraphX Hardcoded Limits (Source Code Analysis)

**Critical:** These limits are hardcoded in ONNX Runtime source, NOT configurable.

### Unsupported Nodes Limit: 10 nodes maximum

From `onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc`:

```cpp
if (unsupported_nodes.size() > 10) {
    return result;  // Rejects ENTIRE graph
}
```

**Impact:** If MIGraphX finds >10 unsupported nodes, it rejects the entire graph without attempting compilation.

### Tensor Size Limit: 300 elements

```cpp
return (std::accumulate(dims.begin(), dims.end(), 1ULL,
        std::multiplies<std::size_t>{}) > 300);
```

**Impact:** Operations (Gemm, MatMul, Conv, LRN, AveragePool, MaxPool) with tensors >300 elements fall back to CPU.

### Model Complexity Examples

| Model | Nodes | Result |
|-------|------:|--------|
| Bi3D FeatNet | 38 | GPU (6.14ms, 163 FPS) |
| Bi3D SegNet | 27 | GPU (11.98ms, 84 FPS) |
| CREStereo small | 4,647 | CPU fallback (575ms, 1.7 FPS) |
| FoundationStereo | 10,000+ | CPU fallback |

**Pattern:** Models with <50 nodes work, models with >1000 nodes fail.

### Why Graphs Get Rejected

```
# Bi3D logs (works):
[MIGraphX EP] Model Compile: Begin
[MIGraphX EP] Model Compile: Complete (13.38s)
All nodes placed on [MIGraphXExecutionProvider]. Number of nodes: 1

# CREStereo logs (fails):
[No compile messages]
All nodes placed on [CPUExecutionProvider]. Number of nodes: 4730
```

**Smoking gun:** MIGraphX never even attempts compilation for complex graphs.

### Workarounds

1. **Use simpler models** - Bi3D works, CREStereo doesn't
2. **Model simplification** - Fewer refinement iterations, simpler backbones
3. **Graph splitting** - Manual subgraph division (complex, may not help)
4. **Wait for improvements** - AMD Issue #4164 acknowledges compilation issues
5. **Custom HIP kernels** - Bypass ONNX Runtime entirely

---

## Quick Reference

| Task | Command/Code |
|------|--------------|
| Install | `pip install onnxruntime-migraphx` |
| Create session | `ort.InferenceSession(model, providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider'])` |
| Enable FP16 | `'fp16_enable': True` in provider options |
| Check providers | `session.get_providers()` |
| Warmup | Run one inference before timing |
| Profile | `options.enable_profiling = True` |
| Debug | `ort.set_default_logger_severity(0)` |
