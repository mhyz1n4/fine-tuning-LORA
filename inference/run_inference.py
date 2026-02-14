import os
import torch
import json
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset

# Default Configuration
DEFAULT_ADAPTER_PATH = "model"
DEFAULT_DATASET_NAME = "Akhil-Theerthala/Kuvera-PersonalFinance-V2.1"
MAX_SEQ_LENGTH = 2048
DTYPE = None # None for auto detection.
LOAD_IN_4BIT = True

SYSTEM_PROMPT = "You are a professional financial advisor specializing in personal finance. Provide accurate, clear, and helpful advice. When thinking through a problem, be logical and consider all financial implications."

def load_model_and_tokenizer(model_path):
    print(f"Loading model and adapter from: {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, instruction):
    # Match the training prompt format
    prompt_style = "### System:\n{system}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    
    formatted_prompt = prompt_style.format(system=SYSTEM_PROMPT, instruction=instruction)
    
    inputs = tokenizer(
        [formatted_prompt],
        return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs.input_ids,
        attention_mask = inputs.attention_mask,
        max_new_tokens = 1024, # Increased for CoT
        use_cache = True,
    )
    
    response = tokenizer.batch_decode(outputs)
    # Extract only the response part
    full_response = response[0] if isinstance(response, list) else response
    response_text = full_response.split("### Response:\n")[-1].replace(tokenizer.eos_token, "").strip()
    return formatted_prompt, response_text

def main():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned Qwen model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ADAPTER_PATH, help="Path to model/adapter directory")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME, help="Dataset name for benchmarking")
    parser.add_argument("--prompt", type=str, help="Custom prompt for inference")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples from test set")
    parser.add_argument("--output", type=str, default="inference/inference_results.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=3407, help="Seed for splitting (must match training)")
    
    args = parser.parse_args()

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []
    
    # 1. Benchmarking on Dataset test samples
    if args.dataset:
        print(f"Loading dataset {args.dataset} and splitting (5% test)...")
        try:
            full_dataset = load_dataset(args.dataset, split="train")
            # Select 6000 samples and split 5% as in train_finance_unsloth.py
            sub_dataset = full_dataset.shuffle(seed=args.seed).select(range(6000))
            test_dataset = sub_dataset.train_test_split(test_size=0.05, seed=args.seed)["test"]
            
            num_to_take = min(args.num_samples, len(test_dataset))
            samples = test_dataset.select(range(num_to_take))
            
            print(f"Generating responses for {num_to_take} test samples...")
            for i, sample in enumerate(samples):
                query = sample.get("query", "")
                ground_truth = sample.get("response", "")
                
                print(f"Sample {i+1}: {query[:50]}...")
                full_prompt, model_response = generate_response(model, tokenizer, query)
                
                results.append({
                    "source": "test_dataset",
                    "input_prompt": full_prompt,
                    "instruction": query,
                    "ground_truth": ground_truth,
                    "model_output": model_response
                })
        except Exception as e:
            print(f"Error loading/processing dataset: {e}")

    # 2. User Prompt
    if args.prompt:
        print(f"Generating response for user prompt: {args.prompt}")
        model_response = generate_response(model, tokenizer, args.prompt)
        results.append({
            "source": "user_input",
            "instruction": args.prompt,
            "expected_response": None,
            "model_response": model_response
        })
    else:
        # Example if no prompt provided
        example_prompt = "What are the best strategies for long-term wealth creation?"
        print(f"No prompt provided. Running example: {example_prompt}")
        model_response = generate_response(model, tokenizer, example_prompt)
        results.append({
            "source": "example",
            "instruction": example_prompt,
            "expected_response": None,
            "model_response": model_response
        })

    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nInference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()