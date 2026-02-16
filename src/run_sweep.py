"""
Sweep runner for experiments and table generation.

Runs multiple experiments and generates summary tables.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import OmegaConf

from .utils import load_config, save_json, load_json, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment sweep")

    parser.add_argument(
        "--sweep_config",
        type=str,
        required=True,
        help="Path to sweep config file",
    )
    parser.add_argument(
        "--model_configs",
        type=str,
        nargs="+",
        default=["configs/model/qwen2.5-3b.yaml"],
        help="Model config files to sweep over",
    )
    parser.add_argument(
        "--data_configs",
        type=str,
        nargs="+",
        default=[
            "configs/data/chemprot.yaml",
            "configs/data/ddi.yaml",
            "configs/data/gad.yaml",
        ],
        help="Data config files to sweep over",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only generate tables from existing results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )

    return parser.parse_args()


def get_model_name(model_config_path: str) -> str:
    """Extract model name from config path."""
    return Path(model_config_path).stem


def get_dataset_name(data_config_path: str) -> str:
    """Extract dataset name from config path."""
    return Path(data_config_path).stem


def run_experiment(
    exp_config_path: str,
    model_config_path: str,
    data_config_path: str,
    variant_config: Optional[Dict[str, Any]],
    output_dir: str,
    variant_name: str,
    use_synthetic: bool = False,
    dry_run: bool = False,
    epochs: int = 3,
    batch_size: int = 8,
) -> Optional[str]:
    """
    Run a single experiment.

    Returns:
        Path to results directory
    """
    model_name = get_model_name(model_config_path)
    dataset_name = get_dataset_name(data_config_path)

    run_output_dir = Path(output_dir) / f"{variant_name}_{model_name}_{dataset_name}"

    # Build command
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", exp_config_path,
        "--model_config", model_config_path,
        "--data_config", data_config_path,
        "--output_dir", str(run_output_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]

    if use_synthetic:
        cmd.append("--use_synthetic")

    # Add variant-specific overrides
    if variant_config:
        # Write temporary config with variant settings
        temp_config_path = run_output_dir / "variant_config.yaml"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(variant_config), temp_config_path)
        # Note: In real implementation, you'd merge this with exp_config

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return str(run_output_dir)

    print(f"\n{'='*60}")
    print(f"Running: {variant_name} | {model_name} | {dataset_name}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return str(run_output_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return None


def collect_results(
    output_dir: str,
    variants: List[str],
    model_names: List[str],
    dataset_names: List[str],
) -> pd.DataFrame:
    """Collect results from all experiments."""
    results = []

    for variant in variants:
        for model in model_names:
            for dataset in dataset_names:
                result_dir = Path(output_dir) / f"{variant}_{model}_{dataset}"
                metrics_file = result_dir / "best_metrics.json"

                if metrics_file.exists():
                    metrics = load_json(metrics_file)
                    results.append({
                        "Variant": variant,
                        "Backbone": model,
                        "Dataset": dataset,
                        "Accuracy": metrics.get("accuracy", 0.0),
                        "Macro_F1": metrics.get("macro_f1", 0.0),
                        "Weighted_F1": metrics.get("weighted_f1", 0.0),
                    })
                else:
                    # Placeholder for missing results
                    results.append({
                        "Variant": variant,
                        "Backbone": model,
                        "Dataset": dataset,
                        "Accuracy": None,
                        "Macro_F1": None,
                        "Weighted_F1": None,
                    })

    return pd.DataFrame(results)


def generate_comparison_table(
    output_dir: str,
    methods: List[str] = ["b1_icl", "b2_lora", "b3_loramoe", "pcr_mola"],
    model_names: List[str] = None,
    dataset_names: List[str] = None,
) -> pd.DataFrame:
    """Generate main comparison table."""
    if model_names is None:
        model_names = ["qwen2.5-3b", "qwen2.5-7b", "qwen3-8b", "llama3.1-8b"]
    if dataset_names is None:
        dataset_names = ["chemprot", "ddi", "gad"]

    return collect_results(output_dir, methods, model_names, dataset_names)


def generate_ablation_table(
    output_dir: str,
    ablation_type: str,
    variants: List[str],
    model_names: List[str] = None,
    dataset_names: List[str] = None,
) -> pd.DataFrame:
    """Generate ablation study table."""
    if model_names is None:
        model_names = ["qwen2.5-3b"]  # Default to one model for ablation
    if dataset_names is None:
        dataset_names = ["chemprot", "ddi", "gad"]

    df = collect_results(output_dir, variants, model_names, dataset_names)

    # Add average column
    if not df.empty:
        numeric_cols = ["Accuracy", "Macro_F1", "Weighted_F1"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Pivot to show datasets as columns
        pivot_df = df.pivot_table(
            index=["Variant", "Backbone"],
            columns="Dataset",
            values=["Macro_F1"],
            aggfunc="first",
        )

        # Flatten column names
        pivot_df.columns = [f"{col[1]}" for col in pivot_df.columns]

        # Add average
        pivot_df["Avg"] = pivot_df.mean(axis=1)

        return pivot_df.reset_index()

    return df


def save_table(df: pd.DataFrame, output_path: str, formats: List[str] = ["csv", "md"]):
    """Save table in multiple formats."""
    output_path = Path(output_path)

    for fmt in formats:
        if fmt == "csv":
            df.to_csv(output_path.with_suffix(".csv"), index=False)
        elif fmt == "md":
            # Generate markdown table
            md_content = df.to_markdown(index=False)
            with open(output_path.with_suffix(".md"), "w") as f:
                f.write(md_content)

    print(f"Saved table to {output_path.with_suffix('.csv')} and {output_path.with_suffix('.md')}")


def main():
    args = parse_args()

    # Setup logger
    logger = setup_logger("sweep")

    # Load sweep config
    sweep_config = load_config(args.sweep_config)

    # Extract sweep info
    sweep_name = sweep_config.get("experiment", {}).get("name", "sweep")
    sweep_variants = sweep_config.get("sweep", {}).get("values", [])

    if not sweep_variants:
        logger.error("No sweep variants found in config")
        return

    # Create output directory
    output_dir = Path(args.output_dir) / sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tables directory
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Get model and dataset names
    model_names = [get_model_name(p) for p in args.model_configs]
    dataset_names = [get_dataset_name(p) for p in args.data_configs]

    if not args.skip_training:
        # Run experiments
        logger.info(f"Running sweep: {sweep_name}")
        logger.info(f"Variants: {[v['name'] for v in sweep_variants]}")
        logger.info(f"Models: {model_names}")
        logger.info(f"Datasets: {dataset_names}")

        for variant in sweep_variants:
            variant_name = variant["name"]
            variant_config = {k: v for k, v in variant.items() if k != "name"}

            for model_config in args.model_configs:
                for data_config in args.data_configs:
                    run_experiment(
                        exp_config_path=args.sweep_config,
                        model_config_path=model_config,
                        data_config_path=data_config,
                        variant_config=variant_config,
                        output_dir=str(output_dir),
                        variant_name=variant_name,
                        use_synthetic=args.use_synthetic,
                        dry_run=args.dry_run,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )

    # Generate tables
    logger.info("Generating summary tables...")

    variant_names = [v["name"] for v in sweep_variants]

    # Main results table
    results_df = collect_results(
        str(output_dir),
        variant_names,
        model_names,
        dataset_names,
    )

    if not results_df.empty:
        save_table(results_df, tables_dir / f"{sweep_name}_results")

        # Pivot table for ablation
        ablation_df = generate_ablation_table(
            str(output_dir),
            sweep_name,
            variant_names,
            model_names,
            dataset_names,
        )
        save_table(ablation_df, tables_dir / f"{sweep_name}_ablation")

        # Print summary
        print("\n" + "=" * 80)
        print(f"SWEEP SUMMARY: {sweep_name}")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)

    logger.info(f"Tables saved to {tables_dir}")


if __name__ == "__main__":
    main()
