"""I/O utilities for saving/loading configs and results."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonlines
import yaml
from omegaconf import OmegaConf, DictConfig


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_predictions(
    predictions: List[Dict[str, Any]],
    path: Union[str, Path]
) -> None:
    """Save predictions to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(predictions)


def load_predictions(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file."""
    with jsonlines.open(path, mode="r") as reader:
        return list(reader)


def load_config(
    config_paths: Union[str, List[str], Path, List[Path]],
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load and merge multiple config files with optional overrides.

    Args:
        config_paths: Single path or list of paths to config files
        overrides: Optional dict of overrides to apply

    Returns:
        Merged OmegaConf DictConfig
    """
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]

    configs = []
    for path in config_paths:
        if str(path).endswith(".yaml") or str(path).endswith(".yml"):
            cfg = OmegaConf.load(path)
        else:
            cfg = OmegaConf.create(load_json(path))
        configs.append(cfg)

    # Merge configs in order (later configs override earlier ones)
    merged = OmegaConf.merge(*configs)

    # Apply overrides if provided
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        merged = OmegaConf.merge(merged, override_cfg)

    return merged


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_git_hash() -> str:
    """Get the current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def save_experiment_info(
    output_dir: Union[str, Path],
    config: DictConfig,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save experiment configuration and environment info."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")

    # Save experiment info
    info = {
        "git_hash": get_git_hash(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    if extra_info:
        info.update(extra_info)

    save_json(info, output_dir / "experiment_info.json")
