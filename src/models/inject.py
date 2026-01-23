"""
Injection utilities for integrating PCR-MoLA into backbone models.

Handles:
- Locating FFN/MLP modules in different architectures (Llama, Qwen)
- Wrapping FFN modules to add PCR-MoLA residual
- Managing entity indices flow through the model
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .pcr_mola import PCRMoLAConfig, PCRMoLAModel, PCRMoLALayer, PairEncoder


# FFN module patterns for different architectures
FFN_PATTERNS = {
    "llama": r"model\.layers\.(\d+)\.mlp",
    "qwen2": r"model\.layers\.(\d+)\.mlp",
    "qwen": r"transformer\.h\.(\d+)\.mlp",
    "mistral": r"model\.layers\.(\d+)\.mlp",
    "gemma": r"model\.layers\.(\d+)\.mlp",
    "phi": r"model\.layers\.(\d+)\.mlp",
}

# Block patterns for getting hidden states before FFN
BLOCK_PATTERNS = {
    "llama": r"model\.layers\.(\d+)",
    "qwen2": r"model\.layers\.(\d+)",
    "qwen": r"transformer\.h\.(\d+)",
    "mistral": r"model\.layers\.(\d+)",
    "gemma": r"model\.layers\.(\d+)",
    "phi": r"model\.layers\.(\d+)",
}


def get_ffn_module_pattern(architecture: str) -> str:
    """Get FFN module regex pattern for the given architecture."""
    arch_lower = architecture.lower()
    for key, pattern in FFN_PATTERNS.items():
        if key in arch_lower:
            return pattern
    # Default pattern
    return r"model\.layers\.(\d+)\.mlp"


def get_block_pattern(architecture: str) -> str:
    """Get block module regex pattern for the given architecture."""
    arch_lower = architecture.lower()
    for key, pattern in BLOCK_PATTERNS.items():
        if key in arch_lower:
            return pattern
    return r"model\.layers\.(\d+)"


class PCRMoLAWrapper(nn.Module):
    """
    Wrapper that adds PCR-MoLA residual to FFN output.

    Wraps the original FFN module and adds:
    ffn_out' = ffn(x) + pcr_mola_layer(x, entity_indices)

    The entity indices are passed via a context manager or stored attribute.
    """

    def __init__(
        self,
        original_ffn: nn.Module,
        pcr_mola_layer: PCRMoLALayer,
        layer_idx: int,
    ):
        super().__init__()
        self.original_ffn = original_ffn
        self.pcr_mola_layer = pcr_mola_layer
        self.layer_idx = layer_idx

        # Entity indices will be set externally before forward
        self._entity_info: Optional[Dict[str, torch.Tensor]] = None

    def set_entity_info(
        self,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Set entity information for the current forward pass."""
        self._entity_info = {
            "e1_indices": e1_indices,
            "e2_indices": e2_indices,
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
            "attention_mask": attention_mask,
        }

    def clear_entity_info(self):
        """Clear entity information after forward pass."""
        self._entity_info = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with PCR-MoLA residual.

        Args:
            hidden_states: Input hidden states to FFN

        Returns:
            FFN output plus PCR-MoLA delta
        """
        # Original FFN forward
        ffn_out = self.original_ffn(hidden_states, *args, **kwargs)

        # Add PCR-MoLA delta if entity info is available
        if self._entity_info is not None:
            delta = self.pcr_mola_layer(
                hidden_states,
                self._entity_info["e1_indices"],
                self._entity_info["e2_indices"],
                self._entity_info["e1_mask"],
                self._entity_info["e2_mask"],
                self._entity_info["attention_mask"],
            )
            ffn_out = ffn_out + delta

        return ffn_out


class PCRMoLAInjectedModel(nn.Module):
    """
    Model with PCR-MoLA injected into FFN layers.

    Manages:
    - Original backbone model
    - PCR-MoLA model with wrapped FFN layers
    - Entity info propagation during forward
    - Balance loss computation
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        pcr_mola_model: PCRMoLAModel,
        wrapped_ffns: List[PCRMoLAWrapper],
    ):
        super().__init__()
        self.backbone = backbone
        self.pcr_mola_model = pcr_mola_model
        self.wrapped_ffns = wrapped_ffns
        self.config = pcr_mola_model.config

    def set_entity_info(
        self,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Broadcast entity info to all wrapped FFN layers."""
        for wrapper in self.wrapped_ffns:
            wrapper.set_entity_info(
                e1_indices, e2_indices, e1_mask, e2_mask, attention_mask
            )

    def clear_entity_info(self):
        """Clear entity info from all wrapped FFN layers."""
        for wrapper in self.wrapped_ffns:
            wrapper.clear_entity_info()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        e1_indices: Optional[torch.Tensor] = None,
        e2_indices: Optional[torch.Tensor] = None,
        e1_mask: Optional[torch.Tensor] = None,
        e2_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with PCR-MoLA.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            e1_indices, e2_indices: (batch_size, max_entity_len)
            e1_mask, e2_mask: (batch_size, max_entity_len)
            labels: (batch_size, seq_len) - for causal LM loss

        Returns:
            Model outputs with additional balance_loss
        """
        # Set entity info for all wrapped layers
        if e1_indices is not None:
            self.set_entity_info(
                e1_indices, e2_indices, e1_mask, e2_mask, attention_mask
            )

        # Reset balance stats
        self.pcr_mola_model.reset_balance_stats()

        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # Clear entity info
        self.clear_entity_info()

        # Compute balance loss
        balance_loss = self.pcr_mola_model.compute_balance_loss()

        # Add balance loss to outputs
        if hasattr(outputs, "loss") and outputs.loss is not None:
            total_loss = outputs.loss + self.config.balance_loss_coef * balance_loss
            outputs.total_loss = total_loss
            outputs.lm_loss = outputs.loss
            outputs.balance_loss = balance_loss
        else:
            outputs.balance_loss = balance_loss

        return outputs

    def generate(self, *args, **kwargs):
        """Generation interface - delegates to backbone."""
        return self.backbone.generate(*args, **kwargs)

    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable parameter counts."""
        pcr_mola_params = self.pcr_mola_model.get_trainable_params()
        backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        return {
            "pcr_mola": pcr_mola_params,
            "backbone_trainable": backbone_trainable,
            "total_trainable": pcr_mola_params + backbone_trainable,
        }


def find_ffn_modules(
    model: PreTrainedModel,
    architecture: str,
) -> Dict[int, Tuple[str, nn.Module]]:
    """
    Find all FFN modules in the model.

    Returns:
        Dict mapping layer index to (full_name, module)
    """
    pattern = get_ffn_module_pattern(architecture)
    ffn_modules = {}

    for name, module in model.named_modules():
        match = re.match(pattern, name)
        if match:
            layer_idx = int(match.group(1))
            ffn_modules[layer_idx] = (name, module)

    return ffn_modules


def inject_pcr_mola(
    model: PreTrainedModel,
    config: PCRMoLAConfig,
    architecture: str = "llama",
    freeze_backbone: bool = True,
) -> PCRMoLAInjectedModel:
    """
    Inject PCR-MoLA into a pretrained model.

    Args:
        model: Pretrained backbone model
        config: PCR-MoLA configuration
        architecture: Model architecture name
        freeze_backbone: Whether to freeze backbone parameters

    Returns:
        PCRMoLAInjectedModel with injected PCR-MoLA layers
    """
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Find FFN modules
    ffn_modules = find_ffn_modules(model, architecture)
    num_layers = len(ffn_modules)

    if num_layers == 0:
        raise ValueError(
            f"No FFN modules found for architecture '{architecture}'. "
            f"Check the FFN_PATTERNS configuration."
        )

    # Update config with actual number of layers
    config.num_layers = num_layers

    # Create PCR-MoLA model
    pcr_mola_model = PCRMoLAModel(config)

    # Move to same device/dtype as backbone
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pcr_mola_model = pcr_mola_model.to(device=device, dtype=dtype)

    # Wrap each FFN module
    wrapped_ffns = []
    for layer_idx in sorted(ffn_modules.keys()):
        name, original_ffn = ffn_modules[layer_idx]
        pcr_mola_layer = pcr_mola_model.get_layer(layer_idx)

        # Create wrapper
        wrapper = PCRMoLAWrapper(original_ffn, pcr_mola_layer, layer_idx)

        # Replace original FFN with wrapper
        # Navigate to parent module and replace
        name_parts = name.split(".")
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name_parts[-1], wrapper)

        wrapped_ffns.append(wrapper)

    # Create injected model
    injected_model = PCRMoLAInjectedModel(model, pcr_mola_model, wrapped_ffns)

    return injected_model


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """Get hidden size from model config."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        return model.config.d_model
    elif hasattr(model.config, "n_embd"):
        return model.config.n_embd
    else:
        raise ValueError("Cannot determine hidden size from model config")


def get_model_num_layers(model: PreTrainedModel) -> int:
    """Get number of layers from model config."""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        return model.config.n_layer
    elif hasattr(model.config, "num_layers"):
        return model.config.num_layers
    else:
        raise ValueError("Cannot determine number of layers from model config")


def create_pcr_mola_from_backbone(
    model: PreTrainedModel,
    architecture: str = "llama",
    router_dim: int = 256,
    rank: int = 8,
    alpha_scale: float = 32.0,
    top_k: int = 2,
    balance_loss_coef: float = 0.01,
    n_experts_low: int = 2,
    n_experts_mid: int = 4,
    n_experts_high: int = 8,
    pair_features: str = "full",
    router_dropout: float = 0.1,
    freeze_backbone: bool = True,
) -> PCRMoLAInjectedModel:
    """
    Convenience function to create PCR-MoLA injected model from backbone.

    Automatically extracts hidden_size and num_layers from backbone config.
    """
    hidden_size = get_model_hidden_size(model)
    num_layers = get_model_num_layers(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    config = PCRMoLAConfig(
        hidden_size=hidden_size,
        router_dim=router_dim,
        rank=rank,
        alpha_scale=alpha_scale,
        top_k=top_k,
        balance_loss_coef=balance_loss_coef,
        num_layers=num_layers,
        n_experts_low=n_experts_low,
        n_experts_mid=n_experts_mid,
        n_experts_high=n_experts_high,
        pair_features=pair_features,
        router_dropout=router_dropout,
        device=str(device),
        dtype=dtype,
    )

    return inject_pcr_mola(model, config, architecture, freeze_backbone)
