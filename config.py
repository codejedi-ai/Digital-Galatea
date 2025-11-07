"""Configuration loading module"""
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_config(config_path="models.yaml"):
    """Load model configuration from YAML file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"✓ Model configuration loaded from {config_path}")
            return config
        else:
            logging.warning(f"⚠ Model configuration file {config_path} not found, using defaults")
            return None
    except Exception as e:
        logging.error(f"✗ Error loading model configuration: {e}")
        return None

# Load configuration at module level
MODEL_CONFIG = load_model_config()

