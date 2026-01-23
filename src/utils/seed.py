"""Random seed utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic mode for CUDA
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For newer PyTorch versions
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some operations may not have deterministic implementations

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_generator(seed: int) -> torch.Generator:
    """Get a torch Generator with the specified seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
