#!/bin/bash

# Ensure the output directory exists

# Loop through all .onnx files in the input directory
for file in onnx-sim/kaoyu/*.onnx; do
    # Check if files exist
    if [ -f "$file" ]; then
        # Extract the filename without the path
        # Run onnxsim  and save to output directory with the same filename
        python -m onnxsim  "$file" "$file"
        echo "onnxsim $file"
    else
        echo "No ONNX files found in onnx/kaoyu/"
        exit 1
    fi
done

echo "Optimization complete!"