import os
import torch
from unsloth import FastLanguageModel
import argparse

def merge_and_save(model_dir, output_dir, format="16bit"):
    print(f"Loading model and adapters from {model_dir}...")
    
    # Load the model and adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    if format == "16bit":
        print(f"Merging and saving to 16-bit Float... (Best for vLLM/Triton)")
        model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")
    elif format == "4bit":
        print(f"Saving to 4-bit (Fastest inference)...")
        model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_4bit")
    elif format.startswith("gguf"):
        # e.g. format="gguf_q4_k_m"
        quant_method = format.split("_")[1:]
        quant_method = "_".join(quant_method) if quant_method else "q4_k_m"
        print(f"Converting to GGUF format ({quant_method})...")
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = quant_method)
    
    print(f"Model successfully saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and convert model.")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory containing the adapter weights")
    parser.add_argument("--output_dir", type=str, default="merged_model", help="Directory to save the merged model")
    parser.add_argument("--format", type=str, default="16bit", choices=["16bit", "4bit", "gguf_q4_k_m", "gguf_q5_k_m"], help="Output format")
    
    args = parser.parse_args()
    merge_and_save(args.model_dir, args.output_dir, args.format)
