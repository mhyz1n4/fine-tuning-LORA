"""
Standard training script using Hugging Face Transformers and PEFT.
"""
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from util import load_yaml_config
from dataloader import load_and_process_dataset
from logger import project_logger as logger

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using LoRA.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B", help="Hugging Face model ID")
    parser.add_argument("--dataset_name", type=str, default="data/toy_train.json", help="Hugging Face dataset ID or local path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for all)")
    parser.add_argument("--config_path", type=str, default="./LORA_peft.yaml", help="Path to YAML config file for LoRA")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU training (slow, not recommended for large models)")
    return parser.parse_args()

def main():
    args = get_args()

    # Load LoRA config from YAML
    lora_yaml_config = load_yaml_config(args.config_path).get("lora_config", {})
    
    # Determine device
    if args.use_cpu:
        device_map = "cpu"
        use_quantization = False
        logger.info("Training on CPU. Quantization disabled.")
    elif torch.cuda.is_available():
        device_map = "auto"
        use_quantization = True
        logger.info("Training on GPU (CUDA) with 4-bit quantization.")
    elif torch.backends.mps.is_available():
        device_map = "auto" # or {"": "mps"}
        use_quantization = False # bitsandbytes usually doesn't work on MPS
        logger.info("Training on MPS (Mac GPU). Quantization disabled.")
    else:
        device_map = "cpu"
        use_quantization = False
        logger.info("Training on CPU. Quantization disabled.")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # 2. Load Model
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
    }

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # Prepare model for training
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    # Map string task type to Enum if necessary, though LoraConfig handles strings well often.
    # To be safe and strict, let's map it if needed, but LoraConfig takes str or TaskType.
    # We will pass the dictionary unpacked.
    
    # Ensure task_type is properly handled if it's a string in yaml matching TaskType enum
    if "task_type" in lora_yaml_config and lora_yaml_config["task_type"] == "CAUSAL_LM":
        lora_yaml_config["task_type"] = TaskType.CAUSAL_LM

    peft_config = LoraConfig(
        **lora_yaml_config
    )

    # 4. Training Arguments (using SFTConfig for trl compatibility)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit" if use_quantization else "adamw_torch",
        save_steps=25,
        logging_steps=25,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        use_cpu=args.use_cpu,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=256,
        packing=False,
    )
    
    # Enable mixed precision if on GPU
    if not args.use_cpu and torch.cuda.is_available():
        sft_config.fp16 = True

    # 5. Dataset
    # Check if dataset_name is a local file (e.g., .json) or a HF dataset
    if args.dataset_name.endswith(".json") and os.path.exists(args.dataset_name):
        logger.info(f"Loading local dataset from {args.dataset_name}")
        dataset = load_and_process_dataset(args.dataset_name)
    else:
        logger.info(f"Loading dataset from Hugging Face: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="train")

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    # 7. Train
    logger.info("Starting training...")
    trainer.train()

    # 8. Save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()