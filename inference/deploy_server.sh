#!/bin/bash

echo "1. Exporting model to 16-bit merged weights..."
./venv/bin/python inference/export_for_deployment.py --model_dir model --output_dir inference/deployed_model

echo "2. Cleaning up any old containers..."
docker compose down

echo "3. Starting vLLM OpenAI-Compatible Server..."
docker compose up -d

echo "--------------------------------------------------------"
echo "Deployment started!"
echo "Check logs with: docker logs -f qwen_vllm_deployment"
echo "API Endpoint: http://localhost:8000/v1/chat/completions"
echo "--------------------------------------------------------"