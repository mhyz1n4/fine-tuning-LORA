from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import os

# Paths
model_path = "./inference/deployed_model"
save_path = "./model/qwen_gptq_8bit"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# 2. Preparation for GPTQ
# We'll use 8-bit as requested. GPTQ works well with 8-bit.
quantize_config = BaseQuantizeConfig(
    bits=8,
    group_size=128,
    desc_act=False,
)

# 3. Load Model
# Note: GPTQ requires a dataset for calibration to maintain accuracy.
# For simplicity in this script, we'll use a small dummy dataset, 
# but for production, you'd use 'wikitext2' or 'c4'.
print("Loading model for GPTQ quantization...")
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 4. Quantize (using a few dummy samples to initialize weights)
# In a real scenario, replace this with meaningful text.
examples = [
    tokenizer("The stock market today showed interesting trends in the technology sector.")
]
print("Starting GPTQ Quantization (8-bit)...")
model.quantize(examples)

# 5. Save
print(f"Saving GPTQ model to {save_path}...")
model.save_quantized(save_path)
tokenizer.save_pretrained(save_path)
print("Done!")
