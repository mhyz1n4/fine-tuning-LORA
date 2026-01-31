"""
Training script using Unsloth for QLoRA fine-tuning.
"""
import argparse
import torch
import os
import sys
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Add parent directory to path to import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import project_logger as logger

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using Unsloth.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B-Instruct", help="Model ID")
    parser.add_argument("--dataset_path", type=str, default="data/toy_train.json", help="Path to local dataset JSON")
    parser.add_argument("--output_dir", type=str, default="./unsloth_results", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    return parser.parse_args()

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"Instruction: {instruction}\nInput: {input}\nResponse: {output}"
        texts.append(text)
    return { "text" : texts }

def main():
    args = get_args()

    # 1. Load Model and Tokenizer
    # Try FP16 first, if OOM, try 8-bit
    logger.info(f"Loading model: {args.model_id}...")
    
    try:
        logger.info("Attempting to load in FP16/BF16...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_length,
            dtype=None, # Auto-detect (Float16 or Bfloat16)
            load_in_4bit=False,
            trust_remote_code=False,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning("OOM detected with FP16. Retrying with 8-bit quantization...")
            torch.cuda.empty_cache()
            try:
                # Note: unsloth from_pretrained passes kwargs to transformers if not 4bit
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=args.model_id,
                    max_seq_length=args.max_seq_length,
                    dtype=None,
                    load_in_4bit=False,
                    load_in_8bit=True,
                    trust_remote_code=False,
                )
            except Exception as e_inner:
                logger.error("Failed to load in 8-bit as well.")
                raise e_inner
        else:
            raise e

    # 2. Add LoRA adapters
    # Unsloth handles target_modules automatically
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # Optimized GC re-enabled
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load and Format Dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {args.dataset_path}. Please run data/generate_toy_data.py first.")
        
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1, 
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    logger.info(f"Training complete. Duration: {trainer_stats.metrics['train_runtime']} seconds")

    # 7. Save
    logger.info(f"Saving LoRA adapters to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
