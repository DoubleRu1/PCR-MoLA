"""Logging utilities for PCR-MoLA."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf


def setup_logger(
    name: str = "pcr_mola",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_str: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Default format
    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(logger: logging.Logger, config: DictConfig, title: str = "Configuration") -> None:
    """Log configuration in a readable format."""
    logger.info(f"{'='*60}")
    logger.info(f"{title}")
    logger.info(f"{'='*60}")
    config_str = OmegaConf.to_yaml(config, resolve=True)
    for line in config_str.split("\n"):
        if line.strip():
            logger.info(line)
    logger.info(f"{'='*60}")


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log metrics in a formatted way."""
    step_str = f"Step {step}: " if step is not None else ""
    prefix_str = f"[{prefix}] " if prefix else ""

    metrics_str = " | ".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in metrics.items()
    )
    logger.info(f"{prefix_str}{step_str}{metrics_str}")


def log_trainable_params(logger: logging.Logger, model: Any) -> Dict[str, int]:
    """Log and return trainable parameter statistics."""
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_ratio = 100 * trainable_params / total_params if total_params > 0 else 0

    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
    }
