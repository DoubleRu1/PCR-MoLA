"""
PCR-MoLA: Entity Pair-Conditioned Routing with Layer-wise Adaptive Expert Allocation.

Core implementation of the PCR-MoLA module including:
- Entity pair encoder with 4-feature representation
- Per-layer router with Top-K selection
- LoRA experts with efficient batch computation
- Balance loss for expert utilization
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


@dataclass
class PCRMoLAConfig:
    """Configuration for PCR-MoLA module."""

    # Hidden dimension of backbone model
    hidden_size: int = 4096

    # Router projection dimension
    router_dim: int = 256

    # LoRA rank for experts
    rank: int = 8

    # Alpha scale for LoRA
    alpha_scale: float = 32.0

    # Top-K routing
    top_k: int = 2

    # Balance loss coefficient
    balance_loss_coef: float = 0.01

    # Number of layers in backbone
    num_layers: int = 32

    # Layer-wise expert allocation
    # [0, L/3): n_experts_low, [L/3, 2L/3): n_experts_mid, [2L/3, L): n_experts_high
    n_experts_low: int = 2
    n_experts_mid: int = 4
    n_experts_high: int = 8

    # Pair feature mode
    # "full": [h1; h2; h1*h2; h1-h2] (4d)
    # "no_diff": [h1; h2; h1*h2] (3d)
    # "no_interaction": [h1; h2; h1-h2] (3d)
    # "context_only": use last token hidden state only
    pair_features: str = "full"

    # Router dropout
    router_dropout: float = 0.1

    # Device
    device: str = "cuda"

    # Dtype
    dtype: torch.dtype = torch.bfloat16

    def get_num_experts(self, layer_idx: int) -> int:
        """Get number of experts for a given layer index."""
        l1 = self.num_layers // 3
        l2 = 2 * self.num_layers // 3

        if layer_idx < l1:
            return self.n_experts_low
        elif layer_idx < l2:
            return self.n_experts_mid
        else:
            return self.n_experts_high

    def get_pair_feature_dim(self) -> int:
        """Get input dimension for pair encoder based on feature mode."""
        if self.pair_features == "context_only":
            return self.hidden_size
        elif self.pair_features in ["no_diff", "no_interaction"]:
            return 3 * self.hidden_size
        else:  # "full"
            return 4 * self.hidden_size


class PairEncoder(nn.Module):
    """
    Entity Pair Encoder that computes pair representation from entity hidden states.

    Supports multiple feature modes:
    - full: [h_e1; h_e2; h_e1 ⊙ h_e2; h_e1 - h_e2]
    - no_diff: [h_e1; h_e2; h_e1 ⊙ h_e2]
    - no_interaction: [h_e1; h_e2; h_e1 - h_e2]
    - context_only: h_last (last token, same for all entity pairs in sentence)
    """

    def __init__(self, config: PCRMoLAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.router_dim = config.router_dim
        self.pair_features = config.pair_features

        input_dim = config.get_pair_feature_dim()

        # Shared projection: W_p with ReLU activation
        self.projection = nn.Linear(input_dim, config.router_dim, bias=True)
        self.dropout = nn.Dropout(config.router_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entity pair representation.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - layer input hidden states
            e1_indices: (batch_size, max_entity_len) - entity 1 token indices
            e2_indices: (batch_size, max_entity_len) - entity 2 token indices
            e1_mask: (batch_size, max_entity_len) - entity 1 valid mask
            e2_mask: (batch_size, max_entity_len) - entity 2 valid mask
            attention_mask: (batch_size, seq_len) - optional attention mask

        Returns:
            pair_repr: (batch_size, router_dim) - pair representation for routing
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if self.pair_features == "context_only":
            # Use last valid token hidden state
            if attention_mask is not None:
                # Find last valid position
                last_pos = attention_mask.sum(dim=1).long() - 1  # (batch_size,)
                last_pos = last_pos.clamp(min=0, max=seq_len - 1)
                # Gather last token hidden states
                idx = last_pos.view(-1, 1, 1).expand(-1, 1, hidden_dim)
                p_raw = hidden_states.gather(1, idx).squeeze(1)  # (batch_size, hidden_dim)
            else:
                p_raw = hidden_states[:, -1, :]  # (batch_size, hidden_dim)
        else:
            # Pool entity hidden states
            h_e1 = self._pool_entity(hidden_states, e1_indices, e1_mask)  # (batch_size, hidden_dim)
            h_e2 = self._pool_entity(hidden_states, e2_indices, e2_mask)  # (batch_size, hidden_dim)

            if self.pair_features == "full":
                # [h_e1; h_e2; h_e1 ⊙ h_e2; h_e1 - h_e2]
                p_raw = torch.cat([
                    h_e1,
                    h_e2,
                    h_e1 * h_e2,
                    h_e1 - h_e2,
                ], dim=-1)
            elif self.pair_features == "no_diff":
                # [h_e1; h_e2; h_e1 ⊙ h_e2]
                p_raw = torch.cat([h_e1, h_e2, h_e1 * h_e2], dim=-1)
            elif self.pair_features == "no_interaction":
                # [h_e1; h_e2; h_e1 - h_e2]
                p_raw = torch.cat([h_e1, h_e2, h_e1 - h_e2], dim=-1)
            else:
                raise ValueError(f"Unknown pair_features mode: {self.pair_features}")

        # Project to router dimension with ReLU
        p = F.relu(self.projection(p_raw))
        p = self.dropout(p)

        return p

    def _pool_entity(
        self,
        hidden_states: torch.Tensor,
        entity_indices: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool entity hidden states."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        max_entity_len = entity_indices.shape[1]

        # Expand indices for gathering
        indices_expanded = entity_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        indices_expanded = indices_expanded.clamp(0, seq_len - 1)

        # Gather hidden states at entity positions
        entity_hidden = torch.gather(hidden_states, dim=1, index=indices_expanded)

        # Masked mean pooling
        mask_expanded = entity_mask.unsqueeze(-1).float()
        entity_sum = (entity_hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1.0)

        return entity_sum / count


class LoRAExpert(nn.Module):
    """
    Single LoRA expert with low-rank decomposition.

    E_k(x) = (alpha / r) * W_up @ (W_down @ x)

    Where:
    - W_down: (r, d) - Gaussian initialized
    - W_up: (d, r) - Zero initialized
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha_scale: float,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha_scale = alpha_scale
        self.scaling = alpha_scale / rank

        # W_down: projects from hidden_size to rank
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        # W_up: projects from rank to hidden_size
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            delta: (batch_size, seq_len, hidden_size)
        """
        return self.scaling * self.up_proj(self.down_proj(x))


class PCRMoLALayer(nn.Module):
    """
    PCR-MoLA layer with entity pair-conditioned routing and multiple LoRA experts.

    Components:
    - Router: W_r @ p -> expert logits
    - Top-K selection with softmax renormalization
    - LoRA experts with efficient batched computation
    - Balance loss accumulation
    """

    def __init__(self, config: PCRMoLAConfig, layer_idx: int, pair_encoder: PairEncoder):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pair_encoder = pair_encoder

        self.hidden_size = config.hidden_size
        self.num_experts = config.get_num_experts(layer_idx)
        self.top_k = min(config.top_k, self.num_experts)
        self.rank = config.rank
        self.alpha_scale = config.alpha_scale
        self.scaling = self.alpha_scale / self.rank

        # Router: W_r (router_dim -> num_experts)
        self.router = nn.Linear(config.router_dim, self.num_experts, bias=False)

        # Experts: store as parameter tensors for efficient batched computation
        # W_down: (num_experts, rank, hidden_size)
        # W_up: (num_experts, hidden_size, rank)
        self.expert_down = nn.Parameter(
            torch.empty(self.num_experts, self.rank, self.hidden_size)
        )
        self.expert_up = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.rank)
        )

        # Initialize experts
        self._init_experts()

        # Balance loss statistics (set during forward, used for loss computation)
        self.balance_stats: Optional[Dict[str, torch.Tensor]] = None

    def _init_experts(self):
        """Initialize expert weights."""
        # W_down: Kaiming/He initialization
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.expert_down[i], a=math.sqrt(5))
        # W_up: Zero initialization
        nn.init.zeros_(self.expert_up)

    def forward(
        self,
        hidden_states: torch.Tensor,
        e1_indices: torch.Tensor,
        e2_indices: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute PCR-MoLA output delta.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - FFN input
            e1_indices, e2_indices: (batch_size, max_entity_len) - entity indices
            e1_mask, e2_mask: (batch_size, max_entity_len) - entity masks
            attention_mask: (batch_size, seq_len)

        Returns:
            delta: (batch_size, seq_len, hidden_size) - residual to add to FFN output
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Step 1: Compute pair representation using shared pair encoder
        # p: (batch_size, router_dim)
        pair_repr = self.pair_encoder(
            hidden_states, e1_indices, e2_indices, e1_mask, e2_mask, attention_mask
        )

        # Step 2: Router logits and softmax
        # g: (batch_size, num_experts)
        router_logits = self.router(pair_repr)
        router_probs = F.softmax(router_logits, dim=-1)

        # Step 3: Top-K selection
        # top_k_probs: (batch_size, top_k)
        # top_k_indices: (batch_size, top_k)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalize top-k probabilities
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Step 4: Compute balance loss statistics
        self._compute_balance_stats(router_probs, top_k_indices)

        # Step 5: Efficient batched expert computation
        # Gather selected expert weights
        # expert_down: (num_experts, rank, hidden_size) -> (batch_size, top_k, rank, hidden_size)
        # expert_up: (num_experts, hidden_size, rank) -> (batch_size, top_k, hidden_size, rank)
        idx_down = top_k_indices.view(batch_size, self.top_k, 1, 1).expand(
            -1, -1, self.rank, self.hidden_size
        )
        idx_up = top_k_indices.view(batch_size, self.top_k, 1, 1).expand(
            -1, -1, self.hidden_size, self.rank
        )

        selected_down = torch.gather(
            self.expert_down.unsqueeze(0).expand(batch_size, -1, -1, -1),
            dim=1,
            index=idx_down,
        )  # (batch_size, top_k, rank, hidden_size)

        selected_up = torch.gather(
            self.expert_up.unsqueeze(0).expand(batch_size, -1, -1, -1),
            dim=1,
            index=idx_up,
        )  # (batch_size, top_k, hidden_size, rank)

        # Step 6: Compute expert outputs with einsum
        # x: (batch_size, seq_len, hidden_size)
        # W_down: (batch_size, top_k, rank, hidden_size)
        # z = x @ W_down^T -> (batch_size, seq_len, top_k, rank)
        z = einsum(
            hidden_states, selected_down,
            "b t d, b k r d -> b t k r"
        )

        # W_up: (batch_size, top_k, hidden_size, rank)
        # y = z @ W_up^T -> (batch_size, seq_len, top_k, hidden_size)
        y = einsum(
            z, selected_up,
            "b t k r, b k d r -> b t k d"
        )

        # Step 7: Weighted sum with top-k weights
        # weights: (batch_size, top_k) -> (batch_size, 1, top_k, 1)
        weights_expanded = top_k_weights.view(batch_size, 1, self.top_k, 1)

        # delta: (batch_size, seq_len, hidden_size)
        delta = (y * weights_expanded).sum(dim=2)

        # Apply scaling
        delta = delta * self.scaling

        return delta

    def _compute_balance_stats(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        """
        Compute balance loss statistics for this layer.

        Args:
            router_probs: (batch_size, num_experts) - full softmax probabilities
            top_k_indices: (batch_size, top_k) - selected expert indices
        """
        batch_size = router_probs.shape[0]

        # f_k: fraction of samples where expert k is in top-K
        # Create one-hot mask for selected experts
        expert_mask = torch.zeros(
            batch_size, self.num_experts,
            device=router_probs.device, dtype=router_probs.dtype
        )
        for k in range(self.top_k):
            expert_mask.scatter_(1, top_k_indices[:, k:k+1], 1.0)

        # f_k: (num_experts,) - average selection frequency
        f_k = expert_mask.mean(dim=0)

        # alpha_mean: (num_experts,) - average router probability
        alpha_mean = router_probs.mean(dim=0)

        self.balance_stats = {
            "f_k": f_k,  # (num_experts,)
            "alpha_mean": alpha_mean,  # (num_experts,)
            "num_experts": self.num_experts,
        }


class PCRMoLAModel(nn.Module):
    """
    Complete PCR-MoLA model wrapper.

    Manages:
    - Shared pair encoder across layers
    - Per-layer PCRMoLALayer modules
    - Total balance loss computation
    """

    def __init__(self, config: PCRMoLAConfig):
        super().__init__()
        self.config = config

        # Shared pair encoder
        self.pair_encoder = PairEncoder(config)

        # Per-layer PCR-MoLA modules
        self.layers = nn.ModuleList([
            PCRMoLALayer(config, layer_idx, self.pair_encoder)
            for layer_idx in range(config.num_layers)
        ])

    def compute_balance_loss(self) -> torch.Tensor:
        """
        Compute total balance loss across all layers.

        L_bal^(l) = N^(l) * Σ_k f_k^(l) * α_mean_k^(l)
        L_bal = (1/L) * Σ_l L_bal^(l)

        Returns:
            balance_loss: Scalar tensor
        """
        total_loss = 0.0
        num_layers_with_stats = 0

        for layer in self.layers:
            if layer.balance_stats is not None:
                stats = layer.balance_stats
                f_k = stats["f_k"]
                alpha_mean = stats["alpha_mean"]
                num_experts = stats["num_experts"]

                # L_bal^(l) = N^(l) * Σ_k f_k * alpha_mean_k
                layer_loss = num_experts * (f_k * alpha_mean).sum()
                total_loss = total_loss + layer_loss
                num_layers_with_stats += 1

        if num_layers_with_stats > 0:
            total_loss = total_loss / num_layers_with_stats

        return total_loss

    def get_layer(self, layer_idx: int) -> PCRMoLALayer:
        """Get PCR-MoLA layer by index."""
        return self.layers[layer_idx]

    def reset_balance_stats(self):
        """Reset balance statistics for all layers."""
        for layer in self.layers:
            layer.balance_stats = None

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_expert_allocation(self):
        """Print layer-wise expert allocation."""
        print("Layer-wise Expert Allocation:")
        print("-" * 40)
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i:2d}: {layer.num_experts} experts")
        print("-" * 40)
        total_experts = sum(layer.num_experts for layer in self.layers)
        print(f"  Total: {total_experts} experts")


def create_pcr_mola(
    hidden_size: int,
    num_layers: int,
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
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> PCRMoLAModel:
    """
    Factory function to create PCR-MoLA model.

    Args:
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        router_dim: Router projection dimension
        rank: LoRA rank
        alpha_scale: LoRA alpha
        top_k: Number of experts to select
        balance_loss_coef: Balance loss coefficient
        n_experts_low/mid/high: Expert counts for layer ranges
        pair_features: Feature mode for pair encoder
        router_dropout: Dropout for router
        device: Device to place model
        dtype: Parameter dtype

    Returns:
        PCRMoLAModel instance
    """
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
        device=device,
        dtype=dtype,
    )

    model = PCRMoLAModel(config)
    model = model.to(device=device, dtype=dtype)

    return model
