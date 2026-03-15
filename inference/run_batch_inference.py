import json
import requests
import argparse
import time
import os
from datasets import load_dataset

SYSTEM_PROMPT = "You are a professional financial advisor specializing in personal finance. Provide accurate, clear, and helpful advice. When thinking through a problem, be logical and consider all financial implications."

def test_vllm_api(instruction, model_name="/model"):
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True
    }
    
    start_time = time.time()
    first_token_time = None
    full_content = ""
    
    try:
        # Use streaming to capture TTFT
        response = requests.post(url, json=payload, timeout=300, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                
                chunk = json.loads(data_str)
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                
                if content and first_token_time is None:
                    first_token_time = time.time()
                
                full_content += content
        
        end_time = time.time()
        
        # Calculate metrics
        # total_latency: Total time (in seconds) from the moment the request is sent until the full response is received.
        # This reflects the end-to-end user experience for a single request.
        total_latency = end_time - start_time
        
        # ttft (Time to First Token): Time (in seconds) from the request start until the first token is received.
        # This is a critical metric for "perceived latency" in interactive applications (how fast the AI 'starts' talking).
        ttft = (first_token_time - start_time) if first_token_time else 0
        
        # generation_time: The time spent strictly on generating tokens after the first one has arrived.
        # Helps isolate the model's throughput from the initial overhead (like prompt processing/prefill).
        generation_time = end_time - first_token_time if first_token_time else 0
        
        # Estimate tokens (approx 1.3 tokens per word for English)
        # Since we don't have the exact tokenizer here, we use a standard heuristic to approximate token count.
        word_count = len(full_content.split())
        estimated_tokens = int(word_count * 1.3)
        
        # tps (Tokens Per Second): The generation throughput.
        # Measures how many tokens the model produces per second during the generation phase.
        tps = estimated_tokens / generation_time if generation_time > 0 else 0
        
        # tpot (Time Per Token): The average latency to generate a single token after the first one.
        # Useful for understanding the cost of generating longer sequences.
        tpot = generation_time / estimated_tokens if estimated_tokens > 0 else 0
        
        return {
            "content": full_content,
            "total_latency": total_latency,
            "ttft": ttft,
            "generation_time": generation_time,
            "tokens": estimated_tokens,
            "tps": tps,
            "tpot": tpot
        }
        
    except Exception as e:
        print(f"Error connecting to server for instruction '{instruction[:20]}...': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Batch inference on deployed Qwen model via vLLM with metrics.")
    parser.add_argument("--dataset", type=str, default="Akhil-Theerthala/Kuvera-PersonalFinance-V2.1", help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="inference/finetuned_results.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=3407, help="Seed for split")
    
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}...")
    full_dataset = load_dataset(args.dataset, split="train")
    sub_dataset = full_dataset.shuffle(seed=args.seed).select(range(6000))
    test_dataset = sub_dataset.train_test_split(test_size=0.05, seed=args.seed)["test"]
    
    num_to_take = min(args.num_samples, len(test_dataset))
    samples = test_dataset.select(range(num_to_take))
    
    results = []
    metrics_summary = {
        "ttft": [],
        "tps": [],
        "tpot": [],
        "latency": []
    }
    
    print(f"Starting batch inference for {num_to_take} samples...")
    
    for i, sample in enumerate(samples):
        query = sample.get("query", "")
        ground_truth = sample.get("response", "")
        
        print(f"[{i+1}/{num_to_take}] Processing: {query[:50]}...")
        
        res = test_vllm_api(query)
        
        if res:
            results.append({
                "instruction": query,
                "ground_truth": ground_truth,
                "model_output": res["content"],
                "metrics": {
                    "total_latency": res["total_latency"],
                    "ttft": res["ttft"],
                    "tps": res["tps"],
                    "tpot": res["tpot"],
                    "tokens": res["tokens"]
                }
            })
            metrics_summary["ttft"].append(res["ttft"])
            metrics_summary["tps"].append(res["tps"])
            metrics_summary["tpot"].append(res["tpot"])
            metrics_summary["latency"].append(res["total_latency"])
            
            print(f"   Done. TTFT: {res['ttft']:.2f}s, TPS: {res['tps']:.2f}, Latency: {res['total_latency']:.2f}s")
        else:
            print(f"   Failed.")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    
    if results:
        avg = lambda x: sum(x) / len(x)
        print("\n" + "="*50)
        print("COMPREHENSIVE METRICS REPORT")
        print("="*50)
        print(f"Total Samples:            {len(results)}")
        print(f"Avg Total Latency:       {avg(metrics_summary['latency']):.2f} s")
        print(f"Avg Time to First Token: {avg(metrics_summary['ttft']):.2f} s")
        print(f"Avg Tokens Per Second:   {avg(metrics_summary['tps']):.2f} tokens/s")
        print(f"Avg Time Per Token:      {avg(metrics_summary['tpot']):.4f} s/token")
        print("="*50)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
