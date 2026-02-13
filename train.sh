#!/bin/bash

# Ensure the model directory exists
mkdir -p model

echo "Starting QLORA Fine-tuning for Personal Finance..."
echo "Output will be logged to model/training_output.log"

# Run the training script
# Resuming logic is already handled inside the python script
./venv/bin/python QLORA/train_finance_unsloth.py 2>&1 | tee model/training_output.log
