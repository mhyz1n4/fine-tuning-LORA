import json
from datasets import Dataset, load_dataset

def format_alpaca(sample):
    """
    Formats a sample dictionary into a single string for SFT training.
    Format:
    ### Instruction:
    {instruction}
    
    ### Input:
    {input}
    
    ### Response:
    {output}
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}<|endoftext|>"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}<|endoftext|>"
    
    return {"text": text}

def load_and_process_dataset(file_path):
    """
    Loads a JSON file and applies formatting.
    Returns a Hugging Face Dataset object with a 'text' column.
    """
    try:
        # Load local JSON file
        # data_files argument allows loading local files directly with load_dataset
        dataset = load_dataset("json", data_files=file_path, split="train")
        
        # Apply formatting
        processed_dataset = dataset.map(format_alpaca)
        
        return processed_dataset
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Test the loader
    train_path = "data/toy_train.json"
    ds = load_and_process_dataset(train_path)
    if ds:
        print(f"Successfully loaded and processed {len(ds)} samples.")
        print("Sample entry:")
        print(ds[0]['text'])
