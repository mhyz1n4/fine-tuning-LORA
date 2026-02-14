import os
import yaml
import json
import torch
import argparse
import random
from string import Template
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_judge(model_id, data_path, prompt_config_path, output_path, mode):
    # Load prompt configuration
    config = load_config(prompt_config_path)
    system_prompt = config['system_prompt']
    user_prompt_template = Template(config['user_prompt_template'])

    # Load data from inference results
    print(f"Loading inference results from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Initialize model and tokenizer
    print(f"Loading judge model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    results = []

    for entry in data:
        instruction = entry.get('instruction', '')
        ground_truth = entry.get('ground_truth', '')
        model_output = entry.get('model_output', '')
        input_prompt = entry.get('input_prompt', '') # Useful context for judge

        if not instruction or not ground_truth or not model_output:
            print(f"Skipping entry missing required fields: {entry.keys()}")
            continue

        if mode == "pairwise":
            # Randomly assign Ground Truth and Model Output to A and B
            # to avoid positional bias
            order = random.choice(["gt_first", "model_first"])
            if order == "gt_first":
                model_a, model_b = ground_truth, model_output
                mapping = {"Model A": "Ground Truth", "Model B": "Model Output"}
            else:
                model_a, model_b = model_output, ground_truth
                mapping = {"Model A": "Model Output", "Model B": "Ground Truth"}

            user_prompt = user_prompt_template.safe_substitute(
                instruction=instruction,
                ground_truth=ground_truth,
                model_a=model_a,
                model_b=model_b
            )
        else:
            # Default scoring mode
            user_prompt = user_prompt_template.safe_substitute(
                instruction=instruction,
                ground_truth=ground_truth,
                model_output=model_output
            )
            mapping = None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        print(f"Judging item: {instruction[:50]}...")
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        result_entry = {
            "instruction": instruction,
            "ground_truth": ground_truth,
            "model_output": model_output,
            "judgment": response,
        }
        
        if mode == "pairwise":
            result_entry["order"] = order
            result_entry["mapping"] = mapping

        results.append(result_entry)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM as a Judge script.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Judge model ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON file with instruction, ground_truth, and model_output")
    parser.add_argument("--prompt_config", type=str, default="evaluate/llama3_judge_prompt.yaml", help="Path to prompt config YAML")
    parser.add_argument("--output_path", type=str, default="evaluate/results.json", help="Path to save results")
    parser.add_argument("--mode", type=str, choices=["scoring", "pairwise"], default="scoring", help="Evaluation mode")

    args = parser.parse_args()
    
    # Auto-detect mode if default prompt is swapped
    if "pairwise" in args.prompt_config and args.mode == "scoring":
        args.mode = "pairwise"
        print("Detected pairwise prompt, switching to pairwise mode.")

    run_judge(args.model_id, args.data_path, args.prompt_config, args.output_path, args.mode)