"""
Generate summary tables from experiment results.

Produces:
- compare_main.csv/md: Main comparison table (B1/B2/B3/Ours × 4 backbones × 3 datasets)
- ablate_routing.csv/md: Routing condition ablation
- ablate_features.csv/md: Entity pair feature ablation
- ablate_experts_balance.csv/md: Expert number and balance loss ablation
- ablate_layerwise_alloc.csv/md: Layer-wise allocation ablation
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary tables")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--tables_dir",
        type=str,
        default="outputs/tables",
        help="Directory to save tables",
    )
    return parser.parse_args()


def load_metrics(result_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from result directory."""
    # Try different metric file names
    for name in ["best_metrics.json", "metrics.json", "eval_test/metrics.json"]:
        metrics_file = result_dir / name
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                return json.load(f)
    return None


def find_results(
    output_dir: Path,
    pattern: str,
) -> Dict[str, Dict[str, Any]]:
    """Find all result directories matching pattern."""
    results = {}
    for path in output_dir.glob(pattern):
        if path.is_dir():
            metrics = load_metrics(path)
            if metrics:
                results[path.name] = metrics
    return results


def generate_main_comparison_table(output_dir: Path) -> pd.DataFrame:
    """Generate main comparison table."""
    methods = ["b1_icl", "b2_lora", "b3_loramoe", "pcr_mola"]
    method_names = {
        "b1_icl": "B1: ICL",
        "b2_lora": "B2: LoRA",
        "b3_loramoe": "B3: LoRAMoE",
        "pcr_mola": "Ours: PCR-MoLA",
    }

    backbones = ["qwen2.5-3b", "qwen2.5-7b", "qwen3-8b", "llama3.1-8b"]
    datasets = ["chemprot", "ddi", "gad"]

    rows = []
    for method in methods:
        for backbone in backbones:
            row = {
                "Method": method_names.get(method, method),
                "Backbone": backbone,
            }

            for dataset in datasets:
                # Look for result directory
                dir_name = f"{method}_{backbone}_{dataset}"
                result_dir = output_dir / dir_name

                metrics = load_metrics(result_dir)
                if metrics:
                    row[f"{dataset}_Acc"] = f"{metrics.get('accuracy', 0) * 100:.1f}"
                    row[f"{dataset}_F1"] = f"{metrics.get('macro_f1', 0) * 100:.1f}"
                else:
                    row[f"{dataset}_Acc"] = "-"
                    row[f"{dataset}_F1"] = "-"

            rows.append(row)

    return pd.DataFrame(rows)


def generate_ablation_table(
    output_dir: Path,
    ablation_name: str,
    variant_names: List[str],
    variant_display_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Generate ablation study table."""
    datasets = ["chemprot", "ddi", "gad"]

    if variant_display_names is None:
        variant_display_names = {v: v for v in variant_names}

    rows = []
    for variant in variant_names:
        row = {"Variant": variant_display_names.get(variant, variant)}

        f1_values = []
        for dataset in datasets:
            # Look for result directory (try multiple patterns)
            for backbone in ["qwen2.5-3b", "qwen2.5-7b"]:
                dir_name = f"{variant}_{backbone}_{dataset}"
                result_dir = output_dir / ablation_name / dir_name

                metrics = load_metrics(result_dir)
                if metrics:
                    f1 = metrics.get("macro_f1", 0) * 100
                    row[dataset] = f"{f1:.1f}"
                    f1_values.append(f1)
                    break
            else:
                row[dataset] = "-"

        # Compute average
        if f1_values:
            row["Avg"] = f"{sum(f1_values) / len(f1_values):.1f}"
        else:
            row["Avg"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path, title: str = ""):
    """Save table as CSV and Markdown."""
    # CSV
    df.to_csv(path.with_suffix(".csv"), index=False)

    # Markdown
    md_content = ""
    if title:
        md_content += f"# {title}\n\n"
    md_content += df.to_markdown(index=False)

    with open(path.with_suffix(".md"), "w") as f:
        f.write(md_content)

    print(f"Saved: {path.with_suffix('.csv')} and {path.with_suffix('.md')}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Summary Tables")
    print("=" * 60)

    # Main comparison table
    print("\n1. Main Comparison Table")
    main_df = generate_main_comparison_table(output_dir)
    save_table(main_df, tables_dir / "compare_main", "Main Comparison Results")

    # Print preview
    print(main_df.to_string(index=False))

    # Ablation: Routing Condition
    print("\n2. Routing Condition Ablation")
    routing_variants = ["static_lora", "sentence_conditioned", "no_routing", "entity_pair_conditioned"]
    routing_names = {
        "static_lora": "Static LoRA (B1)",
        "sentence_conditioned": "Sentence-conditioned",
        "no_routing": "w/o Routing",
        "entity_pair_conditioned": "Ours",
    }
    routing_df = generate_ablation_table(output_dir, "ablate_routing", routing_variants, routing_names)
    save_table(routing_df, tables_dir / "ablate_routing", "Routing Condition Ablation")
    print(routing_df.to_string(index=False))

    # Ablation: Entity Pair Features
    print("\n3. Entity Pair Feature Ablation")
    feature_variants = ["context_only", "no_difference", "no_interaction", "full_features"]
    feature_names = {
        "context_only": "Context Only",
        "no_difference": "w/o Difference",
        "no_interaction": "w/o Interaction",
        "full_features": "Ours (Full)",
    }
    feature_df = generate_ablation_table(output_dir, "ablate_features", feature_variants, feature_names)
    save_table(feature_df, tables_dir / "ablate_features", "Entity Pair Feature Ablation")
    print(feature_df.to_string(index=False))

    # Ablation: Expert Numbers and Balance Loss
    print("\n4. Expert Number & Balance Loss Ablation")
    expert_variants = ["n_experts_2", "n_experts_4", "n_experts_8", "n_experts_16", "n_experts_8_no_balance"]
    expert_names = {
        "n_experts_2": "N=2",
        "n_experts_4": "N=4",
        "n_experts_8": "N=8",
        "n_experts_16": "N=16",
        "n_experts_8_no_balance": "N=8 (w/o Balance)",
    }
    expert_df = generate_ablation_table(output_dir, "ablate_experts_balance", expert_variants, expert_names)
    save_table(expert_df, tables_dir / "ablate_experts_balance", "Expert Number & Balance Loss Ablation")
    print(expert_df.to_string(index=False))

    # Ablation: Layer-wise Allocation
    print("\n5. Layer-wise Allocation Ablation")
    alloc_variants = ["uniform", "reverse", "high_only", "low_only", "progressive"]
    alloc_names = {
        "uniform": "Uniform",
        "reverse": "Reverse",
        "high_only": "High-only",
        "low_only": "Low-only",
        "progressive": "Ours (Progressive)",
    }
    alloc_df = generate_ablation_table(output_dir, "ablate_layerwise_alloc", alloc_variants, alloc_names)
    save_table(alloc_df, tables_dir / "ablate_layerwise_alloc", "Layer-wise Allocation Ablation")
    print(alloc_df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"All tables saved to: {tables_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
