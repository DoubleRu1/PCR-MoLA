"""Dataset classes for relation extraction with entity span tracking."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .preprocess import (
    preprocess_dataset,
    create_prompt,
    load_label_config,
    create_synthetic_data,
)


def find_entity_token_spans(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    e1_start_marker: str = "<e1>",
    e1_end_marker: str = "</e1>",
    e2_start_marker: str = "<e2>",
    e2_end_marker: str = "</e2>",
) -> Tuple[List[int], List[int]]:
    """
    Find token indices for entities based on markers.

    Returns:
        Tuple of (e1_indices, e2_indices) - lists of token positions
    """
    # Get marker token IDs
    # For subword tokenizers, markers might be split - we use the full marker string
    e1_start_ids = tokenizer.encode(e1_start_marker, add_special_tokens=False)
    e1_end_ids = tokenizer.encode(e1_end_marker, add_special_tokens=False)
    e2_start_ids = tokenizer.encode(e2_start_marker, add_special_tokens=False)
    e2_end_ids = tokenizer.encode(e2_end_marker, add_special_tokens=False)

    def find_sublist(main_list: List[int], sublist: List[int], start_from: int = 0) -> int:
        """Find starting index of sublist in main_list."""
        sublist_len = len(sublist)
        for i in range(start_from, len(main_list) - sublist_len + 1):
            if main_list[i:i + sublist_len] == sublist:
                return i
        return -1

    # Find e1 span
    e1_start_pos = find_sublist(input_ids, e1_start_ids)
    e1_end_pos = find_sublist(input_ids, e1_end_ids, e1_start_pos + len(e1_start_ids) if e1_start_pos >= 0 else 0)

    # Find e2 span
    e2_start_pos = find_sublist(input_ids, e2_start_ids)
    e2_end_pos = find_sublist(input_ids, e2_end_ids, e2_start_pos + len(e2_start_ids) if e2_start_pos >= 0 else 0)

    # Entity tokens are between start marker (exclusive) and end marker (exclusive)
    e1_indices = []
    e2_indices = []

    if e1_start_pos >= 0 and e1_end_pos >= 0:
        e1_token_start = e1_start_pos + len(e1_start_ids)
        e1_token_end = e1_end_pos
        e1_indices = list(range(e1_token_start, e1_token_end))

    if e2_start_pos >= 0 and e2_end_pos >= 0:
        e2_token_start = e2_start_pos + len(e2_start_ids)
        e2_token_end = e2_end_pos
        e2_indices = list(range(e2_token_start, e2_token_end))

    # Fallback: if no entity found, use first/last non-special tokens
    if not e1_indices:
        e1_indices = [1]  # Fallback to position 1
    if not e2_indices:
        e2_indices = [2]  # Fallback to position 2

    return e1_indices, e2_indices


class REDataset(Dataset):
    """
    Relation Extraction Dataset with entity span tracking.

    Supports:
    - Entity marker-based span extraction
    - Prompt formatting for instruction-tuned models
    - Label masking for causal LM training
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        label_config: Dict[str, Any],
        task_description: str,
        max_length: int = 128,
        entity_markers: Optional[Dict[str, str]] = None,
        is_train: bool = True,
    ):
        """
        Args:
            examples: List of preprocessed examples
            tokenizer: HuggingFace tokenizer
            label_config: Label configuration dictionary
            task_description: Task description for prompt
            max_length: Maximum sequence length
            entity_markers: Entity marker strings
            is_train: Whether this is training data (includes labels in prompt)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_config = label_config
        self.task_description = task_description
        self.max_length = max_length
        self.is_train = is_train

        self.entity_markers = entity_markers or {
            "e1_start": "<e1>",
            "e1_end": "</e1>",
            "e2_start": "<e2>",
            "e2_end": "</e2>",
        }

        self.label_list = label_config["label_list"]

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Create prompt
        prompt = create_prompt(
            sentence=example["sentence"],
            label_list=self.label_list,
            task_description=self.task_description,
            label_text=example["label"] if self.is_train else None,
            include_answer=self.is_train,
        )

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Find entity token spans
        e1_indices, e2_indices = find_entity_token_spans(
            input_ids.tolist(),
            self.tokenizer,
            self.entity_markers["e1_start"],
            self.entity_markers["e1_end"],
            self.entity_markers["e2_start"],
            self.entity_markers["e2_end"],
        )

        # Create labels for causal LM (mask everything except answer)
        if self.is_train:
            labels = input_ids.clone()
            # Find "### Answer:" position in the actual prompt
            # The prompt has the format: "### Answer:\n{label}"
            # We need to find the position of "### Answer:" in the tokenized input
            # and mask everything before it
            answer_str = "### Answer:"
            answer_tokens = self.tokenizer.encode(answer_str, add_special_tokens=False)

            # Find where "### Answer:" appears in input_ids
            input_list = input_ids.tolist()
            answer_pos = -1
            for i in range(len(input_list) - len(answer_tokens) + 1):
                if input_list[i:i+len(answer_tokens)] == answer_tokens:
                    answer_pos = i
                    break

            # Mask everything before "### Answer:" (inclusive)
            if answer_pos >= 0:
                labels[:answer_pos + len(answer_tokens)] = -100
            else:
                # Fallback: mask most of the prompt
                labels[:100] = -100

            # Also mask padding
            labels[attention_mask == 0] = -100
        else:
            labels = torch.full_like(input_ids, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "e1_indices": e1_indices,
            "e2_indices": e2_indices,
            "label_id": example["label_id"],
            "label_text": example["label"],
            "example_id": example["id"],
        }


def create_dataloader(
    dataset: REDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[callable] = None,
) -> DataLoader:
    """Create a DataLoader for the RE dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def load_re_dataset(
    dataset_name: str,
    data_dir: Union[str, Path],
    labels_file: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    task_description: str,
    entity_markers: Dict[str, str],
    max_length: int = 128,
    split: str = "train",
    use_synthetic: bool = False,
    synthetic_samples: int = 100,
) -> REDataset:
    """
    Load and create RE dataset.

    Args:
        dataset_name: Name of dataset (chemprot, ddi, gad)
        data_dir: Path to data directory
        labels_file: Path to label config file
        tokenizer: HuggingFace tokenizer
        task_description: Task description for prompt
        entity_markers: Entity marker strings
        max_length: Maximum sequence length
        split: Data split (train/dev/test)
        use_synthetic: Whether to use synthetic data
        synthetic_samples: Number of synthetic samples

    Returns:
        REDataset instance
    """
    label_config = load_label_config(labels_file)

    if use_synthetic:
        examples = create_synthetic_data(dataset_name, synthetic_samples, labels_file)
    else:
        examples = preprocess_dataset(
            dataset_name, data_dir, labels_file, entity_markers, split
        )

    return REDataset(
        examples=examples,
        tokenizer=tokenizer,
        label_config=label_config,
        task_description=task_description,
        max_length=max_length,
        entity_markers=entity_markers,
        is_train=(split == "train"),
    )
