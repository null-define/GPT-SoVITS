#!/bin/bash

# Ensure the output directory exists

# Loop through all .onnx files in the input directory
for file in onnx-patched/kaoyu/*.onnx; do
    # Check if files exist
    if [ -f "$file" ]; then
        # Extract the filename without the path
        # Run onnxoptimizer and save to output directory with the same filename
        python -m onnxoptimizer "$file" "$file"
        echo "Optimized $file"
    else
        echo "No ONNX files found in onnxsim-onnx/kaoyu/"
        exit 1
    fi
done

echo "Optimization complete!"