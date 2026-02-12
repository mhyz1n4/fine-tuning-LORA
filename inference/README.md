# Inference Folder

This folder contains scripts for running inference on the fine-tuned Qwen3-8B model.

## Files
- `run_inference.py`: Main script to load the model/adapter and generate responses.
- `inference_results.json`: Output file where results are saved.

## Usage

To run inference on a few samples from the dataset and a custom prompt:

```bash
./venv/bin/python3 inference/run_inference.py --num_samples 3 --prompt "Your custom question here"
```

### Arguments
- `--model_path`: Path to the adapter or model directory (default: `outputs/checkpoint-1`).
- `--dataset`: Hugging Face dataset name for benchmarking (default: `Akhil-Theerthala/Kuvera-PersonalFinance-V2.1`).
- `--prompt`: A custom user prompt to generate a response for.
- `--num_samples`: Number of samples to take from the dataset for benchmarking (default: 3).
- `--output`: Path to the output JSON file (default: `inference/inference_results.json`).

## Example

```bash
./venv/bin/python3 inference/run_inference.py --prompt "How can I start investing with $1000?"
```
