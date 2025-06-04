import os
import glob
from pathlib import Path
import onnx
from onnx import version_converter


# Define input and output directories
input_dir = "onnx-slim/kaoyu"
output_dir = "onnx-patched/kaoyu"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check if onnxruntime is installed
try:
    import onnxruntime
except ImportError:
    print(
        "Error: onnxruntime is not installed. Please install it with 'pip install onnxruntime'"
    )
    exit(1)

# Get list of .onnx files
onnx_files = glob.glob(os.path.join(input_dir, "*.onnx"))

if not onnx_files:
    print(f"Error: No ONNX files found in {input_dir}")
    exit(1)

# Optimize each model
for file_path in onnx_files:
    # if not "decoder" in file_path:
    #     continue
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)

    model_old = onnx.load(file_path)
    model_new = version_converter.convert_version(model_old, 21)
    print(f"saving op covert for: {output_path}")
    onnx.save(model_new, output_path)

    if "decoder" in output_path:
        print(f"processing quant for: {output_path}")
        from onnxruntime.quantization import (
            matmul_nbits_quantizer,
            quant_utils,
            quantize,
        )

        quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=128,  # 2's exponential and >= 16
            is_symmetric=True,  # if true, quantize to Int4. otherwise, quantize to uint4.
            accuracy_level=4,  # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=(
                "MatMul",
            ),  # specify which op types to quantize
            quant_axes=(
                ("MatMul", 0),
            ),  # specify which axis to quantize for an op type.
        )
        model = quant_utils.load_model_with_shape_infer(Path(output_path))
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            nodes_to_exclude=None,  # specify a list of nodes to exclude from quantization
            nodes_to_include=None,  # specify a list of nodes to force include from quantization
            algo_config=quant_config,
        )
        quant.process()
        quant.model.save_model_to_file(output_path, True)  # save data to external file
        print(f"processing quant for: {output_path} done\n")


print("Optimization complete!")
