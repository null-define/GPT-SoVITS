#!/bin/bash

# Ensure the output directory exists

# Loop through all .onnx files in the input directory
for file in onnx-slim/kaoyu/*.onnx; do
    # Check if files exist
    if [ -f "$file" ]; then
        # Extract the filename without the path
        # Run onnxslim  and save to output directory with the same filename
        python -m onnxslim  "$file" "$file"
        echo "onnxslim $file"
    else
        echo "No ONNX files found in onnx/kaoyu/"
        exit 1
    fi
done

echo "Optimization complete!"