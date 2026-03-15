# Monkeypatch for AutoAWQ compatibility with newer transformers and Qwen3
import transformers.activations
if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

import torch
import torch.nn as nn
from awq.quantize.quantizer import AwqQuantizer

def absolute_init_quant(self, n_samples=128, max_seq_len=512):
    from awq.utils.utils import clear_memory
    from tqdm import tqdm

    model = self.model
    layers = self.awq_model.get_model_layers(model)
    
    # Manually prepare samples with padding to ensure uniform size for torch.cat
    samples = []
    for text in self.calib_data:
        if isinstance(text, str):
            inputs = self.tokenizer(
                text, return_tensors="pt", 
                truncation=True, max_length=max_seq_len,
                padding="max_length"
            )
            samples.append(inputs.input_ids)
        else:
            inputs = self.tokenizer(
                text[self.text_column], return_tensors="pt", 
                truncation=True, max_length=max_seq_len,
                padding="max_length"
            )
            samples.append(inputs.input_ids)
            
    samples = torch.cat(samples, dim=0)

    class FixedCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.ins = []
            self.outs = []
            self.module_kwargs = {}

        def forward(self, x, **kwargs):
            self.ins.append(x.cpu())
            self.outs.append(torch.zeros_like(x).cpu()) # dummy
            self.module_kwargs = kwargs
            raise ValueError("Stop forward")
        
        def __getattr__(self, name):
            if name in ['module', 'ins', 'outs', 'module_kwargs']:
                return super().__getattr__(name)
            return getattr(self.module, name)

    # Wrap the first layer
    layers[0] = FixedCatcher(layers[0])
    
    # Run one forward pass to catch the inputs to the first layer
    device = next(model.parameters()).device
    try:
        model(samples.to(device))
    except ValueError as e:
        if str(e) != "Stop forward":
            raise e
    
    # Get the captured data
    captured_ins = layers[0].ins[0]
    captured_kwargs = layers[0].module_kwargs
    
    # Restore the layer
    layers[0] = layers[0].module
    
    clear_memory()
    return layers, captured_kwargs, captured_ins

AwqQuantizer.init_quant = absolute_init_quant

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

# Paths
model_path = "./inference/deployed_model"
save_path = "./model/qwen_awq_4bit"

os.makedirs(save_path, exist_ok=True)

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Configure AWQ
quant_config = { 
    "zero_point": True, 
    "q_group_size": 128, 
    "w_bit": 4, 
    "version": "GEMM" 
}

# 3. Load Model
# Set device_map to "cuda:0" to avoid meta tensors
print("Loading model for AWQ quantization...")
model = AutoAWQForCausalLM.from_pretrained(
    model_path, 
    low_cpu_mem_usage=True, 
    torch_dtype=torch.float16,
    device_map="cuda:0" 
)

# 4. Custom Calibration Data
calibration_data = [
    "The financial markets are showing significant volatility today due to economic reports.",
    "When analyzing stocks, it is important to look at both technical and fundamental indicators.",
    "Quantitative easing is a monetary policy where a central bank purchases government securities.",
    "The technology sector continues to drive innovation in the global economy.",
    "Diversifying your portfolio is a key strategy for long-term investment success."
]

print("Starting AWQ Quantization (4-bit) with custom calibration data...")
model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)

# 5. Save
print(f"Saving AWQ model to {save_path}...")
model.save_quantized(save_path)
tokenizer.save_pretrained(save_path)
print("Done!")
