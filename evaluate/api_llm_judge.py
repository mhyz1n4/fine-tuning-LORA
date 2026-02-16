import os
import yaml
import json
import argparse
import random
import time
from string import Template
from typing import Optional

# To install: pip install openai google-generativeai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def call_openai(client, model, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return response.choices[0].message.content

def call_gemini(model, system_prompt, user_prompt):
    # Gemini handles system prompts differently in the configuration
    # We combine or use the system_instruction parameter if supported
    full_prompt = f"{system_prompt}

[Evaluation Task]
{user_prompt}"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
        )
    )
    return response.text

def run_api_judge():
    parser = argparse.ArgumentParser(description="API-based LLM as a Judge.")
    parser.add_argument("--provider", type=str, choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", type=str, help="Model ID (e.g., gpt-4o-mini or gemini-1.5-flash)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to inference results JSON")
    parser.add_argument("--prompt_config", type=str, default="evaluate/llama3_judge_pairwise.yaml")
    parser.add_argument("--output_path", type=str, default="evaluate/api_results.json")
    
    args = parser.parse_args()

    # Setup Provider
    client = None
    gemini_model = None
    
    if args.provider == "openai":
        if not OpenAI:
            raise ImportError("Please install openai: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        model_id = args.model or "gpt-4o-mini"
    else:
        if not genai:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        model_id = args.model or "gemini-1.5-flash"
        gemini_model = genai.GenerativeModel(model_id)

    # Load prompt and data
    config = load_config(args.prompt_config)
    system_prompt = config['system_prompt']
    user_prompt_template = Template(config['user_prompt_template'])

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    results = []
    print(f"Starting API evaluation using {args.provider} ({model_id})...")

    for i, entry in enumerate(data):
        instruction = entry.get('instruction', '')
        ground_truth = entry.get('ground_truth', '')
        model_output = entry.get('model_output', '')

        if not instruction or not ground_truth or not model_output:
            continue

        # Randomize order
        order = random.choice(["gt_first", "model_first"])
        if order == "gt_first":
            model_a, model_b = ground_truth, model_output
            mapping = {"Model A": "Ground Truth", "Model B": "Model Output"}
        else:
            model_a, model_b = model_output, ground_truth
            mapping = {"Model A": "Model Output", "Model B": "Ground Truth"}

        user_prompt = user_prompt_template.safe_substitute(
            instruction=instruction,
            model_a=model_a,
            model_b=model_b
        )

        print(f"[{i+1}/{len(data)}] Judging item...")
        
        try:
            if args.provider == "openai":
                judgment = call_openai(client, model_id, system_prompt, user_prompt)
            else:
                judgment = call_gemini(gemini_model, system_prompt, user_prompt)
            
            # Verify it's valid JSON
            json_judgment = json.loads(judgment)
            
            results.append({
                "instruction": instruction,
                "ground_truth": ground_truth,
                "model_output": model_output,
                "judgment": json_judgment,
                "order": order,
                "mapping": mapping,
                "provider": args.provider,
                "model": model_id
            })
        except Exception as e:
            print(f"Error judging item {i+1}: {e}")
            time.sleep(2) # Basic backoff

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output_path}")

if __name__ == "__main__":
    run_api_judge()
