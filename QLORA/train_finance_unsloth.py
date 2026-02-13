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
    instructions = examples["query"]
    cots         = examples.get("chain_of_thought", [""] * len(instructions))
    outputs      = examples["response"]
    texts = []
    
    system_prompt = "You are a professional financial advisor specializing in personal finance. Provide accurate, clear, and helpful advice. When thinking through a problem, be logical and consider all financial implications."

    for instruction, cot, output in zip(instructions, cots, outputs):
        if cot:
            text = f"### System:\n{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Response:\n<think>\n{cot}\n</think>\n{output}"
        else:
            # If no CoT is provided in data, we still use the structure to encourage it during inference
            text = f"### System:\n{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return { "text" : texts }

class TimeEstimationCallback(TrainerCallback):
    def __init__(self):
        self.step_times = []
        self.last_step_time = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_step_time is not None:
            step_time = time.time() - self.last_step_time
            self.step_times.append(step_time)
            
        self.last_step_time = time.time()

        if state.global_step == 10:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            total_steps = state.max_steps if state.max_steps > 0 else (state.num_train_epochs * state.max_steps_per_epoch)
            
            estimated_total = avg_step_time * total_steps
            
            logger.info(f"\n--- Training Estimation (Steady State) ---")
            logger.info(f"Avg time per step (steps 2-10): {avg_step_time:.2f} seconds")
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Estimated total time for {args.num_train_epochs} epoch(s): {timedelta(seconds=int(estimated_total))}")
            logger.info(f"------------------------------------------\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step_time = time.time()

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune with Unsloth using YAML config.")
    parser.add_argument("--config_path", type=str, default="QLORA/QLORA_unsloth_4bit.yaml", help="Path to YAML config")
    parser.add_argument("--estimate_only", action="store_true", help="Run 1 step to estimate time")
    return parser.parse_args()

def main():
    args = get_args()
    config = load_yaml_config(args.config_path)
    
    m_cfg = config["model_config"]
    l_cfg = config["lora_config"]
    t_cfg = config["training_config"]

    # Configure local logging
    os.makedirs(t_cfg["output_dir"], exist_ok=True)
    log_path = os.path.join(t_cfg["output_dir"], "training.log")
    logger.add_file_handler(log_path)

    logger.info(f"Loading model {m_cfg['model_id']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = m_cfg["model_id"],
        max_seq_length = m_cfg["max_seq_length"],
        dtype = None,
        load_in_4bit = m_cfg.get("load_in_4bit", False),
        load_in_8bit = m_cfg.get("load_in_8bit", False),
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
        eval_strategy = t_cfg.get("evaluation_strategy", "steps"),
        eval_steps = t_cfg["eval_steps"],
        report_to = "none",
    )

    if args.estimate_only:
        training_args.max_steps = 10
        logger.info("Running in estimation mode (10 steps for steady state)...")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        dataset_text_field = "text",
        max_seq_length = m_cfg["max_seq_length"],
        dataset_num_proc = 2,
        packing = False, # Disabled packing to save memory
        args = training_args,
        callbacks=[TimeEstimationCallback()]
    )

    logger.info("Starting training...")
    
    # Check for existing checkpoints to resume training
    resume_from_checkpoint = None
    if os.path.exists(t_cfg["output_dir"]):
        checkpoints = [os.path.join(t_cfg["output_dir"], d) for d in os.listdir(t_cfg["output_dir"]) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint = resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving final model to {t_cfg['output_dir']}...")
    model.save_pretrained(t_cfg["output_dir"])
    tokenizer.save_pretrained(t_cfg["output_dir"])
    logger.info("Training and saving complete.")

if __name__ == "__main__":
    main()
