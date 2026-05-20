"""
Utility Functions

Helper functions used across the codebase.
"""

import json
import os
import random
import numpy as np
import torch
from typing import Dict, List, Any


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """Save data to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_percent': 100 * trainable / total if total > 0 else 0,
    }


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def print_gpu_memory():
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
