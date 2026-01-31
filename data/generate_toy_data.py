"""
Script to generate toy datasets for training and testing purposes.
"""
import json
import random
import os
import sys

# Add parent directory to path to import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import project_logger as logger

def generate_entry(idx):
    # Generate simple arithmetic problems
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    op = random.choice(["+", "-", "*"])
    
    if op == "+":
        res = a + b
        instruction = "Add the following numbers."
    elif op == "-":
        res = a - b
        instruction = "Subtract the second number from the first."
    else:
        # Keep multiplication simple for toy data
        a = random.randint(1, 12)
        b = random.randint(1, 12)
        res = a * b
        instruction = "Multiply the two numbers."

    return {
        "id": f"toy_{idx}",
        "instruction": instruction,
        "input": f"{a} {op} {b}",
        "output": str(res)
    }

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Generate Training Data (200 samples)
    train_data = [generate_entry(i) for i in range(200)]
    train_path = os.path.join(output_dir, "toy_train.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    logger.info(f"Generated {len(train_data)} training samples at {train_path}")

    # Generate Testing Data (20 samples)
    test_data = [generate_entry(i) for i in range(200, 220)]
    test_path = os.path.join(output_dir, "toy_test.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    logger.info(f"Generated {len(test_data)} testing samples at {test_path}")

if __name__ == "__main__":
    main()