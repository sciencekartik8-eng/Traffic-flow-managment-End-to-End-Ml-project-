"""
Utility helpers for configuration loading, logging, and seed management.
"""

import os
import random
import logging
from typing import Any, Dict
import numpy as np
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file safely.

    Args:
        config_path: Relative or absolute path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across standard Python, NumPy, and PyTorch (if installed).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logger(name: str = "traffic_flow", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger.

    Args:
        name: Logger name.
        level: Logging severity level.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

