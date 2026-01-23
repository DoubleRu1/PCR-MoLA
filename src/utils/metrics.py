"""Metrics computation for relation extraction evaluation."""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def normalize_label(text: str, label_set: Optional[List[str]] = None) -> str:
    """
    Normalize generated label text for matching.

    Args:
        text: Raw generated text
        label_set: Optional list of valid labels for fuzzy matching

    Returns:
        Normalized label string
    """
    # Basic normalization
    text = text.strip().lower()

    # Remove common prefixes/suffixes
    text = re.sub(r"^(the\s+)?(answer\s+is\s*:?\s*)?", "", text)
    text = re.sub(r"^(label\s*:?\s*)?", "", text)
    text = re.sub(r"\s*\.$", "", text)

    # Remove quotes
    text = text.strip("\"'`")

    # Take first word/phrase if multiple lines
    text = text.split("\n")[0].strip()

    # If label_set provided, try to match
    if label_set:
        text_lower = text.lower()
        # Exact match
        for label in label_set:
            if label.lower() == text_lower:
                return label

        # Partial match (text starts with or contains label)
        for label in label_set:
            if text_lower.startswith(label.lower()) or label.lower() in text_lower:
                return label

        # Fuzzy matching based on first characters
        for label in label_set:
            if text_lower[:min(len(text_lower), 4)] == label.lower()[:min(len(label), 4)]:
                return label

    return text


def compute_metrics(
    predictions: List[str],
    references: List[str],
    label_set: Optional[List[str]] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Compute classification metrics for relation extraction.

    Args:
        predictions: List of predicted labels
        references: List of gold labels
        label_set: Optional list of valid labels
        normalize: Whether to normalize labels before comparison

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - macro_f1: Macro-averaged F1
        - weighted_f1: Weighted-averaged F1
        - macro_precision: Macro-averaged precision
        - macro_recall: Macro-averaged recall
        - per_class: Per-class metrics
        - invalid_count: Number of invalid predictions
    """
    if normalize and label_set:
        predictions = [normalize_label(p, label_set) for p in predictions]
        references = [normalize_label(r, label_set) for r in references]

    # Count invalid predictions
    if label_set:
        label_set_lower = {l.lower() for l in label_set}
        invalid_count = sum(
            1 for p in predictions
            if p.lower() not in label_set_lower
        )
    else:
        invalid_count = 0

    # Compute metrics
    accuracy = accuracy_score(references, predictions)

    # Handle case where some labels may not appear in predictions
    all_labels = sorted(set(references) | set(predictions))

    macro_f1 = f1_score(references, predictions, labels=all_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(references, predictions, labels=all_labels, average="weighted", zero_division=0)
    macro_precision = precision_score(references, predictions, labels=all_labels, average="macro", zero_division=0)
    macro_recall = recall_score(references, predictions, labels=all_labels, average="macro", zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(references, predictions, labels=all_labels, average=None, zero_division=0)
    per_class_precision = precision_score(references, predictions, labels=all_labels, average=None, zero_division=0)
    per_class_recall = recall_score(references, predictions, labels=all_labels, average=None, zero_division=0)

    per_class = {}
    for i, label in enumerate(all_labels):
        support = sum(1 for r in references if r == label)
        per_class[label] = {
            "f1": float(per_class_f1[i]),
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "support": support,
        }

    # Confusion matrix
    cm = confusion_matrix(references, predictions, labels=all_labels)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "invalid_count": invalid_count,
        "total_count": len(predictions),
    }


def aggregate_metrics(
    results: List[Dict[str, Any]],
    metric_keys: List[str] = ["accuracy", "macro_f1", "weighted_f1"],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple runs (e.g., different seeds).

    Args:
        results: List of metric dictionaries
        metric_keys: Keys to aggregate

    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return aggregated
