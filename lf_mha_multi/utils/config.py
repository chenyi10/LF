"""Configuration utilities for loading and managing settings"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Use default config
        default_config = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        config_path = str(default_config)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_nested(config: Dict[str, Any], *keys, default=None):
    """Get nested configuration value
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
