# fine-tuning-LORA

A repo for fine-tunning LLM and SLM models using LORA library.

## System Requirements

To run the training and evaluation scripts in this repository, your system must meet the following requirements:

### Hardware
- **GPU**: NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3080/4080 or better).
- **RAM**: Minimum 16GB System RAM (32GB recommended for loading larger models).
- **Disk**: ~20GB free space for model checkpoints and datasets.

### Software & Drivers
- **NVIDIA Drivers**: Version 535 or newer.
- **CUDA Toolkit**: Version 12.1 or newer (must include `nvcc` for `torch.compile` support).
- **NVIDIA Container Toolkit**: Required if running inside Docker.
- **Python**: 3.9, 3.10, 3.11, or 3.12.

### Installation Details
Ensure `nvcc` is accessible in your PATH:
```bash
nvcc --version
```
If you encounter permission errors with `nvcc`, ensure your user has execution rights to the CUDA binaries or run the environment with appropriate privileges.