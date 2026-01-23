"""Data collator for RE datasets with entity span support."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer


@dataclass
class REDataCollator:
    """
    Data collator for Relation Extraction that handles entity spans.

    Handles:
    - Padding of input_ids, attention_mask, labels
    - Padding of entity indices to fixed length
    - Creation of entity masks for variable-length entity spans
    """

    tokenizer: PreTrainedTokenizer
    max_entity_length: int = 10  # Maximum tokens per entity span
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.

        Args:
            features: List of feature dictionaries from dataset

        Returns:
            Batched tensors including entity indices and masks
        """
        batch_size = len(features)

        # Separate entity indices from other features
        e1_indices_list = [f.pop("e1_indices") for f in features]
        e2_indices_list = [f.pop("e2_indices") for f in features]
        label_ids = [f.pop("label_id") for f in features]
        label_texts = [f.pop("label_text") for f in features]
        example_ids = [f.pop("example_id") for f in features]

        # Stack tensor features
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        # Pad entity indices to fixed length
        e1_indices_padded = torch.zeros(batch_size, self.max_entity_length, dtype=torch.long)
        e2_indices_padded = torch.zeros(batch_size, self.max_entity_length, dtype=torch.long)
        e1_mask = torch.zeros(batch_size, self.max_entity_length, dtype=torch.bool)
        e2_mask = torch.zeros(batch_size, self.max_entity_length, dtype=torch.bool)

        for i, (e1_idx, e2_idx) in enumerate(zip(e1_indices_list, e2_indices_list)):
            # Truncate if too long
            e1_idx = e1_idx[:self.max_entity_length]
            e2_idx = e2_idx[:self.max_entity_length]

            e1_len = len(e1_idx)
            e2_len = len(e2_idx)

            if e1_len > 0:
                e1_indices_padded[i, :e1_len] = torch.tensor(e1_idx, dtype=torch.long)
                e1_mask[i, :e1_len] = True
            if e2_len > 0:
                e2_indices_padded[i, :e2_len] = torch.tensor(e2_idx, dtype=torch.long)
                e2_mask[i, :e2_len] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "e1_indices": e1_indices_padded,
            "e2_indices": e2_indices_padded,
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "label_texts": label_texts,
            "example_ids": example_ids,
        }


@dataclass
class ICLDataCollator:
    """
    Data collator for In-Context Learning (no training).
    Simpler version without label masking.
    """

    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch for ICL inference."""
        batch_size = len(features)

        # Extract metadata
        label_ids = [f.pop("label_id", -1) for f in features]
        label_texts = [f.pop("label_text", "") for f in features]
        example_ids = [f.pop("example_id", i) for i, f in enumerate(features)]

        # Remove entity indices if present
        for f in features:
            f.pop("e1_indices", None)
            f.pop("e2_indices", None)

        # Stack tensors
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "label_texts": label_texts,
            "example_ids": example_ids,
        }


def get_entity_hidden_states(
    hidden_states: torch.Tensor,
    entity_indices: torch.Tensor,
    entity_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract and pool entity hidden states.

    Args:
        hidden_states: (batch_size, seq_len, hidden_dim)
        entity_indices: (batch_size, max_entity_len) - token indices
        entity_mask: (batch_size, max_entity_len) - valid position mask

    Returns:
        Pooled entity representation: (batch_size, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    max_entity_len = entity_indices.shape[1]

    # Expand indices for gathering
    # entity_indices: (batch_size, max_entity_len) -> (batch_size, max_entity_len, hidden_dim)
    indices_expanded = entity_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)

    # Clamp indices to valid range
    indices_expanded = indices_expanded.clamp(0, seq_len - 1)

    # Gather hidden states at entity positions
    # Result: (batch_size, max_entity_len, hidden_dim)
    entity_hidden = torch.gather(hidden_states, dim=1, index=indices_expanded)

    # Apply mask and compute mean pooling
    # entity_mask: (batch_size, max_entity_len) -> (batch_size, max_entity_len, 1)
    mask_expanded = entity_mask.unsqueeze(-1).float()

    # Masked sum
    entity_sum = (entity_hidden * mask_expanded).sum(dim=1)  # (batch_size, hidden_dim)

    # Count valid positions (avoid division by zero)
    count = mask_expanded.sum(dim=1).clamp(min=1.0)  # (batch_size, 1)

    # Mean pooling
    entity_pooled = entity_sum / count

    return entity_pooled
