import argparse
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using Unsloth.")
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen2.5-7B-bnb-4bit", help="Model ID (use unsloth pre-quantized versions for max speed)")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco", help="Hugging Face dataset ID")
    parser.add_argument("--output_dir", type=str, default="./unsloth_results", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length (Unsloth handles up to 128k)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--save_method", type=str, default="lora", choices=["lora", "merged_16bit", "merged_4bit", "gguf"], help="How to save the final model")
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Load Model and Tokenizer (Unsloth optimized)
    # 4-bit pre-quantized models from unsloth are faster to download and load
    print(f"Loading Unsloth model: {args.model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None, # None = auto detection (Float16 for Tesla T4, Bfloat16 for Ampere+)
        load_in_4bit=True,
    )

    # 2. Add LoRA adapters
    # Unsloth handles target_modules automatically
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0, # optimized to 0
        bias="none",    # optimized to none
        use_gradient_checkpointing="unsloth", # optimized GC
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load Dataset
    print(f"Loading dataset: {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train")

    # 4. Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1, # Train on all data (overrides epochs if set to positive value)
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
        packing=False, # Can speed up training for short sequences
        args=training_args,
    )

    # 6. Train
    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Duration: {trainer_stats.metrics['train_runtime']} seconds")

    # 7. Save
    print(f"Saving model to {args.output_dir} with method: {args.save_method}")
    if args.save_method == "lora":
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    elif args.save_method == "merged_16bit":
        model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")
    elif args.save_method == "merged_4bit":
        model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_4bit")
    elif args.save_method == "gguf":
        # Saves to q8_0 by default, can change quantization method
        model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method="q8_0")

if __name__ == "__main__":
    main()