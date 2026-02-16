import os
import torch
from unsloth import FastLanguageModel
import argparse

def export_model(model_dir, output_dir):
    print(f"Loading model and adapters from {model_dir}...")
    
    # Load the model and adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    print(f"Merging and saving to 16-bit BFloat16 for deployment...")
    # This creates a standard Hugging Face model folder
    model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")
    
    print(f"Deployment-ready model saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned model for vLLM/Triton.")
    parser.add_argument("--model_dir", type=str, default="model", help="Path to your adapter weights")
    parser.add_argument("--output_dir", type=str, default="inference/deployed_model", help="Where to save merged model")
    
    args = parser.parse_args()
    export_model(args.model_dir, args.output_dir)
