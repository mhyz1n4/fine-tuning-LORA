# Docker Compose Deployment Makefile

COMPOSE_DIR := deploy_configs
TRITON_CONFIG := $(COMPOSE_DIR)/docker-compose.yml
VLLM_CONFIG := $(COMPOSE_DIR)/docker-compose.vllm.yaml

.PHONY: help deploy-triton stop-triton deploy-vllm stop-vllm quantize-gptq quantize-awq

help:
	@echo "Usage:"
	@echo "  make deploy-triton   - Deploy Triton Inference Server"
	@echo "  make stop-triton     - Stop Triton Inference Server"
	@echo "  make deploy-vllm     - Deploy vLLM Inference Server"
	@echo "  make stop-vllm       - Stop vLLM Inference Server"
	@echo "  make quantize-gptq   - Convert model to GPTQ 8-bit"
	@echo "  make quantize-awq    - Convert model to AWQ 4-bit"

deploy-triton:
	docker compose -f $(TRITON_CONFIG) up -d

stop-triton:
	docker compose -f $(TRITON_CONFIG) down

deploy-vllm:
	docker compose -f $(VLLM_CONFIG) up -d

stop-vllm:
	docker compose -f $(VLLM_CONFIG) down

quantize-gptq:
	. ./venv/bin/activate && pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/ && pip install optimum
	./venv/bin/python quantize_gptq.py

quantize-awq:
	. ./venv/bin/activate && pip install autoawq
	./venv/bin/python quantize_awq.py
