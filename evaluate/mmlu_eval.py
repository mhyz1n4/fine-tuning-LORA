import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def format_mmlu_prompt(example, subject_name):
    prompt = f"The following are multiple choice questions (with answers) about {subject_name.replace('_', ' ')}.\n\n"
    prompt += f"Question: {example['question']}\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer: "
    return prompt

def run_comprehensive_eval(model_id, subjects, total_limit):
    print(f"Loading model: {model_id}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    all_y_true = []
    all_y_pred = []
    detailed_results = []
    
    # Calculate samples per subject to reach total_limit
    samples_per_subject = max(1, total_limit // len(subjects))
    
    print(f"Starting evaluation on {len(subjects)} subjects, ~{samples_per_subject} samples each.")

    for subject in subjects:
        try:
            dataset = load_dataset("tasksource/mmlu", subject, split="test", trust_remote_code=True)
            subset_size = min(samples_per_subject, len(dataset))
            dataset = dataset.select(range(subset_size))
        except Exception as e:
            print(f"Skipping subject {subject} due to error: {e}")
            continue

        for example in tqdm(dataset, desc=f"Eval {subject}"):
            prompt = format_mmlu_prompt(example, subject)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5, # Slightly increased to capture potential whitespace/context
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated tokens
            full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Basic validation of prediction format
            y_pred_raw = full_response.upper()
            y_pred = y_pred_raw[0] if len(y_pred_raw) > 0 and y_pred_raw[0] in "ABCD" else "INVALID"
            y_true = chr(65 + example['answer'])
            
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            
            detailed_results.append({
                "subject": subject,
                "question": example['question'],
                "prompt": prompt,
                "model_response": full_response,
                "predicted": y_pred,
                "ground_truth": y_true,
                "is_correct": y_pred == y_true
            })

    # Filter out invalid predictions for metric calculation
    valid_indices = [i for i, p in enumerate(all_y_pred) if p != "INVALID"]
    y_true_valid = [all_y_true[i] for i in valid_indices]
    y_pred_valid = [all_y_pred[i] for i in valid_indices]
    
    invalid_count = len(all_y_pred) - len(y_pred_valid)

    # Industry Standard Metrics
    # We use 'macro' averaging for a balanced view across all answer classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_valid, 
        y_pred_valid, 
        labels=["A", "B", "C", "D"], 
        average='macro',
        zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        y_true_valid, 
        y_pred_valid, 
        labels=["A", "B", "C", "D"], 
        average=None,
        zero_division=0
    )

    accuracy = np.mean([1 if p == t else 0 for p, t in zip(all_y_pred, all_y_true)])

    metrics = {
        "model_id": model_id,
        "total_samples": len(all_y_true),
        "invalid_responses": invalid_count,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "per_class_metrics": {
            label: {"precision": p, "recall": r, "f1": f}
            for label, p, r, f in zip(["A", "B", "C", "D"], p_class, r_class, f1_class)
        },
        "detailed_results": detailed_results
    }

    print(f"\n--- Comprehensive MMLU Results ---")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"Invalid:   {invalid_count} samples")
    print(f"----------------------------------\n")

    with open("evaluate/mmlu_comprehensive_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    # Expanded list of subjects covering various domains
    MMLU_SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics", "econometrics",
        "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "world_religions"
    ]

    parser = argparse.ArgumentParser(description="Comprehensive MMLU Evaluation")
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen3-8B-bnb-4bit", help="Model ID")
    parser.add_argument("--total_limit", type=int, default=500, help="Total samples to evaluate")
    
    args = parser.parse_args()
    run_comprehensive_eval(args.model_id, MMLU_SUBJECTS, args.total_limit)