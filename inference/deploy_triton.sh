#!/bin/bash

echo "1. Exporting model to 16-bit merged weights..."
./venv/bin/python inference/export_for_deployment.py --model_dir model --output_dir inference/deployed_model

echo "2. Cleaning up any old containers..."
docker-compose down

echo "3. Starting Triton Inference Server with vLLM backend..."
docker-compose up -d

echo "--------------------------------------------------------"
echo "Deployment started!"
echo "Check logs with: docker logs -f triton_qwen_deployment"
echo "Endpoint: http://localhost:8000/v2/models/qwen_model/generate"
echo "--------------------------------------------------------"
