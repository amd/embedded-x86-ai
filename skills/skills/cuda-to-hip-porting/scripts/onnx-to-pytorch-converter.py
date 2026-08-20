#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
ONNX to PyTorch Conversion Script (S63)
Converts ONNX models to native PyTorch for cross-platform ROCm/CUDA execution.

Usage:
    python onnx-to-pytorch-converter.py input.onnx output.pt [--validate]
"""
import os
import sys
import argparse
import numpy as np

def check_dependencies():
    """Check and report missing dependencies."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnx2torch
    except ImportError:
        missing.append("onnx2torch")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def extract_onnx_weights(onnx_path):
    """Extract weights from ONNX model to numpy dict."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    return weights


def get_onnx_info(onnx_path):
    """Get ONNX model information."""
    import onnx

    model = onnx.load(onnx_path)

    info = {
        'opset_version': model.opset_import[0].version if model.opset_import else 'unknown',
        'ir_version': model.ir_version,
        'producer': model.producer_name,
        'inputs': [],
        'outputs': [],
        'num_nodes': len(model.graph.node),
        'num_initializers': len(model.graph.initializer),
    }

    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        info['inputs'].append({'name': inp.name, 'shape': shape})

    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        info['outputs'].append({'name': out.name, 'shape': shape})

    # Check for custom ops
    standard_ops = set()
    custom_ops = set()
    for node in model.graph.node:
        if node.domain in ['', 'ai.onnx', 'com.microsoft']:
            standard_ops.add(node.op_type)
        else:
            custom_ops.add(f"{node.domain}::{node.op_type}")

    info['standard_ops'] = len(standard_ops)
    info['custom_ops'] = list(custom_ops)

    return info


def convert_onnx_to_pytorch(onnx_path, output_path=None, simplify_first=True):
    """Convert ONNX model to PyTorch."""
    import torch
    import onnx
    from onnx2torch import convert

    print(f"\nLoading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # Get model info
    info = get_onnx_info(onnx_path)
    print(f"  Opset version: {info['opset_version']}")
    print(f"  Nodes: {info['num_nodes']}")
    print(f"  Inputs: {info['inputs']}")
    print(f"  Custom ops: {info['custom_ops'] if info['custom_ops'] else 'None'}")

    if info['custom_ops']:
        print("\nWARNING: Model contains custom operators!")
        print("Consider decomposing first with tools/decompose_custom_ops.py (S47)")

    # Optionally simplify first
    if simplify_first:
        try:
            from onnxsim import simplify
            print("\nSimplifying ONNX model...")
            onnx_model, check = simplify(onnx_model)
            if check:
                print("  Simplification successful")
            else:
                print("  WARNING: Simplification may have changed model behavior")
        except ImportError:
            print("  onnxsim not installed, skipping simplification")
        except Exception as e:
            print(f"  Simplification failed: {e}")

    # Convert to PyTorch
    print("\nConverting to PyTorch...")
    try:
        pytorch_model = convert(onnx_model)
        pytorch_model.eval()
        print("  Conversion successful!")
    except Exception as e:
        print(f"  ERROR: Conversion failed: {e}")
        print("\nTry decomposing custom ops first:")
        print("  python tools/decompose_custom_ops.py input.onnx decomposed.onnx")
        return None

    # Count parameters
    params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"  Parameters: {params/1e6:.2f}M")

    # Save if output path provided
    if output_path:
        print(f"\nSaving PyTorch model to: {output_path}")
        if output_path.endswith('.pt') or output_path.endswith('.pth'):
            # Save state dict
            torch.save(pytorch_model.state_dict(), output_path)
            print("  Saved as state_dict")
        else:
            # Save as TorchScript
            try:
                scripted = torch.jit.script(pytorch_model)
                scripted.save(output_path)
                print("  Saved as TorchScript")
            except Exception as e:
                print(f"  TorchScript failed: {e}")
                print("  Falling back to state_dict...")
                torch.save(pytorch_model.state_dict(), output_path + ".pth")

    return pytorch_model


def validate_conversion(onnx_path, pytorch_model, num_tests=3):
    """Validate that ONNX and PyTorch models produce same outputs."""
    import torch
    import onnx
    import onnxruntime as ort

    print("\nValidating conversion...")

    # Get input info
    info = get_onnx_info(onnx_path)

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    max_diffs = []
    for i in range(num_tests):
        # Create random input
        inputs = {}
        torch_inputs = []
        for inp in info['inputs']:
            # Replace dynamic dims with reasonable values
            shape = []
            for d in inp['shape']:
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                else:
                    shape.append(1)  # Default batch size

            data = np.random.randn(*shape).astype(np.float32)
            inputs[inp['name']] = data
            torch_inputs.append(torch.from_numpy(data))

        # Run ONNX model
        onnx_outputs = ort_session.run(None, inputs)

        # Run PyTorch model
        pytorch_model.eval()
        with torch.no_grad():
            if len(torch_inputs) == 1:
                pytorch_outputs = pytorch_model(torch_inputs[0])
            else:
                pytorch_outputs = pytorch_model(*torch_inputs)

        if not isinstance(pytorch_outputs, (list, tuple)):
            pytorch_outputs = [pytorch_outputs]

        # Compare outputs
        for j, (onnx_out, pytorch_out) in enumerate(zip(onnx_outputs, pytorch_outputs)):
            diff = np.abs(onnx_out - pytorch_out.numpy()).max()
            max_diffs.append(diff)
            print(f"  Test {i+1}, Output {j}: max diff = {diff:.2e}")

    avg_diff = np.mean(max_diffs)
    max_diff = np.max(max_diffs)

    print(f"\nValidation Summary:")
    print(f"  Average max diff: {avg_diff:.2e}")
    print(f"  Maximum diff: {max_diff:.2e}")

    if max_diff < 1e-4:
        print("  Status: PASS (diffs within tolerance)")
        return True
    elif max_diff < 1e-2:
        print("  Status: WARNING (small numerical differences)")
        return True
    else:
        print("  Status: FAIL (significant differences)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to PyTorch')
    parser.add_argument('input', help='Input ONNX model path')
    parser.add_argument('output', nargs='?', help='Output PyTorch model path')
    parser.add_argument('--validate', action='store_true', help='Validate conversion')
    parser.add_argument('--no-simplify', action='store_true', help='Skip ONNX simplification')
    parser.add_argument('--info-only', action='store_true', help='Only show ONNX model info')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    if not check_dependencies():
        return 1

    # Info only mode
    if args.info_only:
        info = get_onnx_info(args.input)
        print("\nONNX Model Info:")
        print(f"  File: {args.input}")
        print(f"  Size: {os.path.getsize(args.input) / 1e6:.1f} MB")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return 0

    # Convert
    pytorch_model = convert_onnx_to_pytorch(
        args.input,
        args.output,
        simplify_first=not args.no_simplify
    )

    if pytorch_model is None:
        return 1

    # Validate if requested
    if args.validate:
        try:
            import onnxruntime
            success = validate_conversion(args.input, pytorch_model)
            if not success:
                return 1
        except ImportError:
            print("WARNING: onnxruntime not installed, skipping validation")

    print("\nConversion complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
