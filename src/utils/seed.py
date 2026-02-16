"""Random seed utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def get_available_device() -> str:
    """
    Get the best available device (cuda > mps > cpu).

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    # Check environment variables first
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and os.environ.get("MPS_VISIBLE_DEVICES", "") == "":
        # Both not set - use default detection
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        # At least one is explicitly set - respect them
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        mps_visible = os.environ.get("MPS_VISIBLE_DEVICES", "")

        # Check CUDA first
        if cuda_visible and cuda_visible != "":
            if torch.cuda.is_available():
                return "cuda"
        elif mps_visible and mps_visible != "":
            if torch.backends.mps.is_available():
                return "mps"

        # Fallback to CPU if explicitly disabled
        return "cpu"


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

    # MPS also supports manual seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_generator(seed: int) -> torch.Generator:
    """Get a torch Generator with the specified seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
