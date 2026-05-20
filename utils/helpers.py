import os
import shutil
from typing import Optional

def cleanup_directory(directory_path: str, verbose: bool = True) -> bool:
    """Remove directory if it exists"""
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        if verbose:
            print(f"Directory '{directory_path}' removed successfully.")
        return True
    else:
        if verbose:
            print(f"Directory '{directory_path}' does not exist.")
        return False

def ensure_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(directory_path, exist_ok=True)
    
def print_model_summary(model) -> None:
    """Print model summary"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")