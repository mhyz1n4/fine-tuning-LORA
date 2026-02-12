import os
import torch
import json
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset

# Default Configuration
DEFAULT_ADAPTER_PATH = "outputs/checkpoint-1"
DEFAULT_DATASET_NAME = "Akhil-Theerthala/Kuvera-PersonalFinance-V2.1"
MAX_SEQ_LENGTH = 2048
DTYPE = None # None for auto detection.
LOAD_IN_4BIT = True

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
    prompt_style = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    
    inputs = tokenizer(
        [prompt_style.format(instruction)],
        return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs.input_ids,
        attention_mask = inputs.attention_mask,
        max_new_tokens = 512,
        use_cache = True,
    )
    
    response = tokenizer.batch_decode(outputs)
    # Extract only the response part
    response_text = response[0].split("### Response:\n")[-1].replace(tokenizer.eos_token, "").strip()
    return response_text

def main():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned Qwen model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ADAPTER_PATH, help="Path to model/adapter directory")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME, help="Dataset name for benchmarking")
    parser.add_argument("--prompt", type=str, help="Custom prompt for inference")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples from dataset to benchmark")
    parser.add_argument("--output", type=str, default="inference/inference_results.json", help="Output JSON file")
    
    args = parser.parse_args()

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []
    
    # 1. Benchmarking on Dataset samples
    if args.dataset:
        print(f"Loading dataset {args.dataset}...")
        try:
            dataset = load_dataset(args.dataset, split="train")
            # Select some samples
            num_to_take = min(args.num_samples, len(dataset))
            samples = dataset.select(range(num_to_take))
            
            print(f"Generating responses for {num_to_take} dataset samples...")
            for i, sample in enumerate(samples):
                query = sample.get("query", sample.get("instruction", ""))
                expected = sample.get("response", sample.get("output", ""))
                
                print(f"Sample {i+1}: {query[:50]}...")
                model_response = generate_response(model, tokenizer, query)
                
                results.append({
                    "source": "dataset",
                    "instruction": query,
                    "expected_response": expected,
                    "model_response": model_response
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