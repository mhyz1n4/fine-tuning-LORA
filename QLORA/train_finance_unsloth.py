import os
import sys
import torch
import time
import yaml
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datetime import timedelta

# Add parent directory to path to import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import project_logger as logger
from util import load_yaml_config

def formatting_prompts_func(examples):
    instructions = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return { "text" : texts }

class TimeEstimationCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            elapsed_time = time.time() - self.start_time
            total_steps = state.max_steps if state.max_steps > 0 else (state.num_train_epochs * state.max_steps_per_epoch)
            
            estimated_total = elapsed_time * state.max_steps
            
            logger.info(f"\n--- Training Estimation ---")
            logger.info(f"Time for 1 step: {elapsed_time:.2f} seconds")
            logger.info(f"Total steps: {state.max_steps}")
            logger.info(f"Estimated total time for {args.num_train_epochs} epoch(s): {timedelta(seconds=int(estimated_total))}")
            logger.info(f"---------------------------\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune with Unsloth using YAML config.")
    parser.add_argument("--config_path", type=str, default="QLORA/kuvera_config.yaml", help="Path to YAML config")
    parser.add_argument("--estimate_only", action="store_true", help="Run 1 step to estimate time")
    return parser.parse_args()

def main():
    args = get_args()
    config = load_yaml_config(args.config_path)
    
    m_cfg = config["model_config"]
    l_cfg = config["lora_config"]
    t_cfg = config["training_config"]

    logger.info(f"Loading model {m_cfg['model_id']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = m_cfg["model_id"],
        max_seq_length = m_cfg["max_seq_length"],
        dtype = None,
        load_in_4bit = m_cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        **l_cfg
    )

    # Load dataset
    logger.info(f"Loading dataset {t_cfg['dataset_name']}...")
    dataset = load_dataset(t_cfg["dataset_name"], split="train")
    
    dataset = dataset.shuffle(seed=t_cfg["seed"]).select(range(t_cfg["dataset_num_samples"]))
    
    dataset = dataset.train_test_split(test_size=t_cfg["test_size"])
    train_dataset = dataset["train"].map(formatting_prompts_func, batched=True)
    test_dataset = dataset["test"].map(formatting_prompts_func, batched=True)

    training_args = TrainingArguments(
        per_device_train_batch_size = t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = t_cfg["gradient_accumulation_steps"],
        warmup_steps = t_cfg["warmup_steps"],
        num_train_epochs = t_cfg["num_train_epochs"],
        learning_rate = t_cfg["learning_rate"],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = t_cfg["logging_steps"],
        optim = t_cfg["optim"],
        weight_decay = t_cfg["weight_decay"],
        lr_scheduler_type = t_cfg["lr_scheduler_type"],
        seed = t_cfg["seed"],
        output_dir = t_cfg["output_dir"],
        save_steps = t_cfg["save_steps"],
        save_total_limit = t_cfg["save_total_limit"],
        evaluation_strategy = t_cfg["evaluation_strategy"],
        eval_steps = t_cfg["eval_steps"],
        report_to = "none",
    )

    if args.estimate_only:
        training_args.max_steps = 1
        logger.info("Running in estimation mode (1 step)...")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        dataset_text_field = "text",
        max_seq_length = m_cfg["max_seq_length"],
        dataset_num_proc = 2,
        packing = False,
        args = training_args,
        callbacks=[TimeEstimationCallback()]
    )

    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()