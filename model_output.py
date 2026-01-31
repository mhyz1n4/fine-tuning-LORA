"""
Script to export Unsloth LoRA adapters or generate outputs.
"""
import argparse
import os
import torch
from unsloth import FastLanguageModel
from logger import project_logger as logger

def get_args():
    parser = argparse.ArgumentParser(description="Export Unsloth LoRA adapters for vLLM or TensorRT-LLM deployment.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B-Instruct", help="Base model ID")
    parser.add_argument("--lora_path", type=str, default="unsloth_results", help="Path to LoRA adapters")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for exported models")
    parser.add_argument("--format", type=str, choices=["vllm", "tensorrt", "both"], default="both", help="Target deployment format")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for the model")
    return parser.parse_args()

def load_model(model_id, lora_path, max_seq_length):
    logger.info(f"Loading model: {model_id} with adapters from {lora_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path, # Load the adapters directly (Unsloth auto-loads base model)
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True, # Must load in 4bit to merge efficiently
    )
    return model, tokenizer

def export_for_vllm(model, tokenizer, output_path):
    logger.info(f"[vLLM Export] Merging weights and saving to: {output_path}")
    
    # vLLM works best with standard 16-bit Float/Bfloat .safetensors
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method = "merged_16bit",
    )
    logger.log_block([
        f"[vLLM Export] Success! You can serve this model using:",
        f"  vllm serve {output_path} --dtype auto"
    ])

def export_for_tensorrt(model, tokenizer, output_path):
    logger.info(f"[TensorRT-LLM Export] Preparing intermediate HF model at: {output_path}")
    
    # TensorRT-LLM build requires a clean HF 16-bit checkpoint
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method = "merged_16bit",
    )
    
    logger.info(f"[TensorRT-LLM Export] Intermediate model saved.")
    
    logger.log_block([
        "-" * 60,
        "NEXT STEPS FOR TENSORRT-LLM:",
        "TensorRT-LLM requires compilation using the `trtllm-build` tool, usually run inside the NVIDIA container.",
        "\n1. Pull the container:",
        "   docker pull nvcr.io/nvidia/tensorrt-llm:latest",
        "\n2. Run the build command (adjust --output_dir and GPU architecture):",
        f"   trtllm-build --checkpoint_dir {os.path.abspath(output_path)} \\",
        f"                --output_dir {os.path.abspath(output_path)}_engine \\",
        f"                --gemm_plugin float16",
        "-" * 60
    ])

def main():
    args = get_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load once
    model, tokenizer = load_model(args.model_id, args.lora_path, args.max_seq_length)

    # Process vLLM
    if args.format in ["vllm", "both"]:
        vllm_dir = os.path.join(args.output_dir, "vllm_model")
        export_for_vllm(model, tokenizer, vllm_dir)

    # Process TensorRT
    if args.format in ["tensorrt", "both"]:
        trt_dir = os.path.join(args.output_dir, "tensorrt_hf_intermediate")
        export_for_tensorrt(model, tokenizer, trt_dir)

    logger.info(f"All export tasks completed.")

if __name__ == "__main__":
    main()