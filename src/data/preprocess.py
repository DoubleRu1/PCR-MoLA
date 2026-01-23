"""Data preprocessing functions for RE datasets."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset


def load_label_config(labels_file: Union[str, Path]) -> Dict[str, Any]:
    """Load label configuration from JSON file."""
    with open(labels_file, "r", encoding="utf-8") as f:
        return json.load(f)


def insert_entity_markers(
    text: str,
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
    e1_start_marker: str = "<e1>",
    e1_end_marker: str = "</e1>",
    e2_start_marker: str = "<e2>",
    e2_end_marker: str = "</e2>",
) -> str:
    """
    Insert entity markers into text at specified character positions.

    Args:
        text: Original text
        e1_start, e1_end: Character positions for entity 1
        e2_start, e2_end: Character positions for entity 2
        *_marker: Entity marker strings

    Returns:
        Text with entity markers inserted
    """
    # Determine order (which entity comes first)
    if e1_start <= e2_start:
        # e1 comes first
        parts = [
            text[:e1_start],
            e1_start_marker,
            text[e1_start:e1_end],
            e1_end_marker,
            text[e1_end:e2_start],
            e2_start_marker,
            text[e2_start:e2_end],
            e2_end_marker,
            text[e2_end:],
        ]
    else:
        # e2 comes first
        parts = [
            text[:e2_start],
            e2_start_marker,
            text[e2_start:e2_end],
            e2_end_marker,
            text[e2_end:e1_start],
            e1_start_marker,
            text[e1_start:e1_end],
            e1_end_marker,
            text[e1_end:],
        ]

    return "".join(parts)


def create_prompt(
    sentence: str,
    label_list: List[str],
    task_description: str,
    label_text: Optional[str] = None,
    include_answer: bool = True,
) -> str:
    """
    Create instruction prompt for RE task.

    Args:
        sentence: Sentence with entity markers
        label_list: List of possible labels
        task_description: Task description
        label_text: Gold label (for training)
        include_answer: Whether to include answer section

    Returns:
        Formatted prompt string
    """
    labels_str = ", ".join(label_list)

    prompt = f"""### Instruction:
{task_description}
Given the sentence with marked entities <e1>...</e1> and <e2>...</e2>, classify the relation.
Choose one label from: [{labels_str}].
Output ONLY the label.

### Sentence:
{sentence}

### Answer:
"""
    if include_answer and label_text is not None:
        prompt += label_text

    return prompt


def preprocess_chemprot(
    data_dir: Union[str, Path],
    labels_file: Union[str, Path],
    entity_markers: Dict[str, str],
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Preprocess ChemProt dataset.

    Expected data format (TSV or JSON):
    - sentence: text
    - e1_start, e1_end: entity 1 character offsets
    - e2_start, e2_end: entity 2 character offsets
    - label: relation label

    Returns:
        List of preprocessed examples
    """
    data_dir = Path(data_dir)
    label_config = load_label_config(labels_file)

    # Try different file formats
    data_file = None
    for ext in [".json", ".jsonl", ".tsv", ".txt"]:
        candidate = data_dir / f"{split}{ext}"
        if candidate.exists():
            data_file = candidate
            break

    if data_file is None:
        raise FileNotFoundError(f"No data file found for split '{split}' in {data_dir}")

    examples = []

    if str(data_file).endswith(".json"):
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif str(data_file).endswith(".jsonl"):
        raw_data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                raw_data.append(json.loads(line.strip()))
    else:
        # TSV format
        raw_data = []
        with open(data_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                values = line.strip().split("\t")
                raw_data.append(dict(zip(header, values)))

    for item in raw_data:
        # Extract fields (handle different naming conventions)
        sentence = item.get("sentence", item.get("text", ""))
        e1_start = int(item.get("e1_start", item.get("entity1_start", 0)))
        e1_end = int(item.get("e1_end", item.get("entity1_end", 0)))
        e2_start = int(item.get("e2_start", item.get("entity2_start", 0)))
        e2_end = int(item.get("e2_end", item.get("entity2_end", 0)))
        label = item.get("label", item.get("relation", "false"))

        # Insert entity markers
        marked_sentence = insert_entity_markers(
            sentence, e1_start, e1_end, e2_start, e2_end,
            entity_markers["e1_start"], entity_markers["e1_end"],
            entity_markers["e2_start"], entity_markers["e2_end"],
        )

        # Map label to text
        label_id = label_config["label_to_id"].get(str(label), 0)
        label_text = label_config["id_to_label"][str(label_id)]

        examples.append({
            "id": item.get("id", len(examples)),
            "sentence": marked_sentence,
            "label": label_text,
            "label_id": label_id,
            "original_sentence": sentence,
            "e1_text": sentence[e1_start:e1_end],
            "e2_text": sentence[e2_start:e2_end],
        })

    return examples


def preprocess_ddi(
    data_dir: Union[str, Path],
    labels_file: Union[str, Path],
    entity_markers: Dict[str, str],
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Preprocess DDI dataset.
    Similar structure to ChemProt.
    """
    # Use same preprocessing logic as ChemProt
    return preprocess_chemprot(data_dir, labels_file, entity_markers, split)


def preprocess_gad(
    data_dir: Union[str, Path],
    labels_file: Union[str, Path],
    entity_markers: Dict[str, str],
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Preprocess GAD dataset.
    Similar structure to ChemProt.
    """
    # Use same preprocessing logic as ChemProt
    return preprocess_chemprot(data_dir, labels_file, entity_markers, split)


def preprocess_dataset(
    dataset_name: str,
    data_dir: Union[str, Path],
    labels_file: Union[str, Path],
    entity_markers: Dict[str, str],
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Generic preprocessing function that dispatches to dataset-specific preprocessors.
    """
    preprocessors = {
        "chemprot": preprocess_chemprot,
        "ddi": preprocess_ddi,
        "gad": preprocess_gad,
    }

    if dataset_name not in preprocessors:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(preprocessors.keys())}")

    return preprocessors[dataset_name](data_dir, labels_file, entity_markers, split)


def create_synthetic_data(
    dataset_name: str,
    num_samples: int = 100,
    labels_file: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Create synthetic data for testing.
    Useful when actual data is not available.
    """
    import random

    # Default label configs
    default_labels = {
        "chemprot": ["no_relation", "upregulator", "downregulator", "agonist", "antagonist", "substrate"],
        "ddi": ["no_interaction", "mechanism", "effect", "advise", "interaction"],
        "gad": ["not_associated", "associated"],
    }

    labels = default_labels.get(dataset_name, ["no_relation", "related"])

    # Synthetic sentences
    templates = [
        "The compound <e1>{e1}</e1> was found to interact with <e2>{e2}</e2> in the study.",
        "Studies show that <e1>{e1}</e1> affects the expression of <e2>{e2}</e2>.",
        "The drug <e1>{e1}</e1> and <e2>{e2}</e2> were analyzed for potential interactions.",
        "Research indicates <e1>{e1}</e1> may regulate <e2>{e2}</e2> activity.",
        "The relationship between <e1>{e1}</e1> and <e2>{e2}</e2> was investigated.",
    ]

    entity_names = ["DrugA", "DrugB", "ProteinX", "ProteinY", "GeneZ", "CompoundC"]

    examples = []
    for i in range(num_samples):
        template = random.choice(templates)
        e1 = random.choice(entity_names)
        e2 = random.choice([e for e in entity_names if e != e1])
        sentence = template.format(e1=e1, e2=e2)
        label = random.choice(labels)

        examples.append({
            "id": i,
            "sentence": sentence,
            "label": label,
            "label_id": labels.index(label),
            "original_sentence": sentence,
            "e1_text": e1,
            "e2_text": e2,
        })

    return examples
