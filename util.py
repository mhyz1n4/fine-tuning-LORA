"""
General utility functions, such as loading YAML configurations.
"""
import yaml
import os

def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The configuration dictionary.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not a valid YAML.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file: {e}")
