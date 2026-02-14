#!/bin/bash

# Default directories
ADAPTER_DIR="model"
OUTPUT_DIR="final_vllm_model"

echo "Merging adapters from $ADAPTER_DIR and exporting to $OUTPUT_DIR for vLLM/Triton..."

# Run the merge script using the virtual environment
./venv/bin/python QLORA/merge_and_convert.py 
    --model_dir "$ADAPTER_DIR" 
    --output_dir "$OUTPUT_DIR" 
    --format "16bit"

echo "Done. You can now point vLLM or Triton to the '$OUTPUT_DIR' directory."
