"""
Baseline implementations for comparison with PCR-MoLA.

B1: In-Context Learning (ICL) - few-shot prompting
B2: Standard LoRA - static low-rank adaptation
B3: LoRAMoE - token-conditioned routing MoE
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import PreTrainedModel, PreTrainedTokenizer

from einops import einsum


# =============================================================================
# B1: In-Context Learning (ICL)
# =============================================================================

def select_exemplars(
    examples: List[Dict[str, Any]],
    num_exemplars: int = 5,
    strategy: str = "stratified",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select exemplars for in-context learning.

    Args:
        examples: List of training examples
        num_exemplars: Number of exemplars to select
        strategy: "random" or "stratified"
        seed: Random seed

    Returns:
        List of selected exemplars
    """
    random.seed(seed)

    if strategy == "stratified":
        # Group by label
        by_label = {}
        for ex in examples:
            label = ex["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(ex)

        # Sample evenly from each label
        exemplars = []
        labels = list(by_label.keys())
        per_label = max(1, num_exemplars // len(labels))

        for label in labels:
            available = by_label[label]
            n = min(per_label, len(available))
            exemplars.extend(random.sample(available, n))

        # If we need more, sample randomly
        while len(exemplars) < num_exemplars:
            remaining = [ex for ex in examples if ex not in exemplars]
            if not remaining:
                break
            exemplars.append(random.choice(remaining))

        return exemplars[:num_exemplars]
    else:
        # Random selection
        return random.sample(examples, min(num_exemplars, len(examples)))


def create_icl_prompt(
    sentence: str,
    label_list: List[str],
    task_description: str,
    exemplars: List[Dict[str, Any]],
    use_cot: bool = False,
) -> str:
    """
    Create ICL prompt with exemplars.

    Args:
        sentence: Input sentence with entity markers
        label_list: List of valid labels
        task_description: Task description
        exemplars: Few-shot exemplars
        use_cot: Whether to use chain-of-thought

    Returns:
        Formatted ICL prompt
    """
    labels_str = ", ".join(label_list)

    # Build exemplar demonstrations
    demo_parts = []
    for ex in exemplars:
        demo = f"""### Sentence:
{ex['sentence']}

### Answer:
{ex['label']}"""
        demo_parts.append(demo)

    demos = "\n\n".join(demo_parts)

    # Full prompt
    prompt = f"""### Instruction:
{task_description}
Given the sentence with marked entities <e1>...</e1> and <e2>...</e2>, classify the relation.
Choose one label from: [{labels_str}].
Output ONLY the label.

Here are some examples:

{demos}

Now classify the following:

### Sentence:
{sentence}

### Answer:
"""

    return prompt


class ICLModel:
    """
    Wrapper for ICL inference (no training).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        exemplars: List[Dict[str, Any]],
        label_list: List[str],
        task_description: str,
        use_cot: bool = False,
        max_new_tokens: int = 16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.exemplars = exemplars
        self.label_list = label_list
        self.task_description = task_description
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens

        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def predict(self, sentence: str) -> str:
        """Generate prediction for a single sentence."""
        prompt = create_icl_prompt(
            sentence=sentence,
            label_list=self.label_list,
            task_description=self.task_description,
            exemplars=self.exemplars,
            use_cot=self.use_cot,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Longer for ICL context
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated.strip()

    def predict_batch(
        self,
        sentences: List[str],
        batch_size: int = 1,
    ) -> List[str]:
        """Generate predictions for multiple sentences."""
        predictions = []
        for sentence in sentences:
            pred = self.predict(sentence)
            predictions.append(pred)
        return predictions


# =============================================================================
# B2: Standard LoRA
# =============================================================================

def create_lora_model(
    model: PreTrainedModel,
    rank: int = 8,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
) -> PreTrainedModel:
    """
    Create a model with standard LoRA adapters using PEFT.

    Args:
        model: Pretrained backbone model
        rank: LoRA rank
        alpha: LoRA alpha (scaling)
        dropout: LoRA dropout
        target_modules: Modules to apply LoRA (default: q_proj, v_proj)
        modules_to_save: Additional modules to train

    Returns:
        PEFT model with LoRA adapters
    """
    if target_modules is None:
        # Default target modules for attention
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)

    return peft_model


# =============================================================================
# B3: LoRAMoE (Token-Conditioned Routing)
# =============================================================================

class LoRAMoEExpert(nn.Module):
    """Single LoRA expert for LoRAMoE."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.lora_up(self.lora_down(x))


class LoRAMoELayer(nn.Module):
    """
    LoRAMoE layer with token-conditioned routing.

    Each token routes independently to top-K experts based on its hidden state.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        rank: int = 8,
        alpha: float = 32.0,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.top_k = min(top_k, num_experts)
        self.scaling = alpha / rank

        # Router: token hidden state -> expert scores
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.router_dropout = nn.Dropout(dropout)

        # Experts stored as parameter tensors for efficiency
        # down: (num_experts, rank, hidden_size)
        # up: (num_experts, hidden_size, rank)
        self.expert_down = nn.Parameter(
            torch.empty(num_experts, rank, hidden_size)
        )
        self.expert_up = nn.Parameter(
            torch.empty(num_experts, hidden_size, rank)
        )

        self._init_weights()

        # Balance stats
        self.balance_stats: Optional[Dict[str, torch.Tensor]] = None

    def _init_weights(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.expert_down[i], a=math.sqrt(5))
        nn.init.zeros_(self.expert_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            delta: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Flatten for routing
        x_flat = x.view(batch_size * seq_len, hidden_dim)

        # Router: (batch*seq, num_experts)
        router_logits = self.router(self.router_dropout(x_flat))
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute balance stats
        self._compute_balance_stats(router_probs, top_k_indices)

        # Gather selected experts
        # expert_down: (num_experts, rank, hidden_size)
        # selected: (batch*seq, top_k, rank, hidden_size)
        bs = batch_size * seq_len
        idx_down = top_k_indices.view(bs, self.top_k, 1, 1).expand(
            -1, -1, self.rank, hidden_dim
        )
        idx_up = top_k_indices.view(bs, self.top_k, 1, 1).expand(
            -1, -1, hidden_dim, self.rank
        )

        selected_down = torch.gather(
            self.expert_down.unsqueeze(0).expand(bs, -1, -1, -1),
            dim=1,
            index=idx_down,
        )
        selected_up = torch.gather(
            self.expert_up.unsqueeze(0).expand(bs, -1, -1, -1),
            dim=1,
            index=idx_up,
        )

        # Compute expert outputs
        # x_flat: (batch*seq, hidden_size) -> (batch*seq, 1, hidden_size)
        x_expanded = x_flat.unsqueeze(1)

        # z = x @ W_down^T -> (batch*seq, top_k, rank)
        z = einsum(x_expanded, selected_down, "n 1 d, n k r d -> n k r")

        # y = z @ W_up^T -> (batch*seq, top_k, hidden_size)
        y = einsum(z, selected_up, "n k r, n k d r -> n k d")

        # Weighted sum
        weights = top_k_weights.unsqueeze(-1)  # (batch*seq, top_k, 1)
        delta_flat = (y * weights).sum(dim=1)  # (batch*seq, hidden_size)
        delta_flat = delta_flat * self.scaling

        # Reshape back
        delta = delta_flat.view(batch_size, seq_len, hidden_dim)

        return delta

    def _compute_balance_stats(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        """Compute balance loss statistics."""
        n = router_probs.shape[0]

        # Expert selection frequency
        expert_mask = torch.zeros(
            n, self.num_experts,
            device=router_probs.device, dtype=router_probs.dtype
        )
        for k in range(self.top_k):
            expert_mask.scatter_(1, top_k_indices[:, k:k+1], 1.0)

        f_k = expert_mask.mean(dim=0)
        alpha_mean = router_probs.mean(dim=0)

        self.balance_stats = {
            "f_k": f_k,
            "alpha_mean": alpha_mean,
            "num_experts": self.num_experts,
        }


class LoRAMoEWrapper(nn.Module):
    """Wrapper that adds LoRAMoE to a linear layer."""

    def __init__(
        self,
        original_linear: nn.Linear,
        loramoe_layer: LoRAMoELayer,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.loramoe_layer = loramoe_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_linear(x)
        delta = self.loramoe_layer(x)
        return original_out + delta


class LoRAMoEModel(nn.Module):
    """
    Model with LoRAMoE adapters.

    Injects LoRAMoE into attention projection layers (q_proj, v_proj).
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        loramoe_layers: List[LoRAMoELayer],
        wrappers: List[LoRAMoEWrapper],
        balance_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.backbone = backbone
        self.loramoe_layers = nn.ModuleList(loramoe_layers)
        self.wrappers = wrappers
        self.balance_loss_coef = balance_loss_coef

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Reset balance stats
        for layer in self.loramoe_layers:
            layer.balance_stats = None

        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # Compute balance loss
        balance_loss = self._compute_balance_loss()

        if hasattr(outputs, "loss") and outputs.loss is not None:
            total_loss = outputs.loss + self.balance_loss_coef * balance_loss
            outputs.total_loss = total_loss
            outputs.lm_loss = outputs.loss
            outputs.balance_loss = balance_loss
        else:
            outputs.balance_loss = balance_loss

        return outputs

    def _compute_balance_loss(self) -> torch.Tensor:
        total_loss = 0.0
        count = 0

        for layer in self.loramoe_layers:
            if layer.balance_stats is not None:
                stats = layer.balance_stats
                f_k = stats["f_k"]
                alpha_mean = stats["alpha_mean"]
                num_experts = stats["num_experts"]

                layer_loss = num_experts * (f_k * alpha_mean).sum()
                total_loss = total_loss + layer_loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)


def create_loramoe_model(
    model: PreTrainedModel,
    num_experts: int = 4,
    rank: int = 8,
    alpha: float = 32.0,
    top_k: int = 2,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    balance_loss_coef: float = 0.01,
    freeze_backbone: bool = True,
) -> LoRAMoEModel:
    """
    Create a model with LoRAMoE adapters.

    Args:
        model: Pretrained backbone
        num_experts: Number of LoRA experts
        rank: LoRA rank
        alpha: LoRA alpha
        top_k: Top-K routing
        dropout: Router dropout
        target_modules: Modules to apply LoRAMoE
        balance_loss_coef: Balance loss coefficient
        freeze_backbone: Whether to freeze backbone

    Returns:
        LoRAMoEModel instance
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Get hidden size
    hidden_size = model.config.hidden_size

    # Find and wrap target modules
    loramoe_layers = []
    wrappers = []

    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Create LoRAMoE layer
                loramoe = LoRAMoELayer(
                    hidden_size=module.in_features,
                    num_experts=num_experts,
                    rank=rank,
                    alpha=alpha,
                    top_k=top_k,
                    dropout=dropout,
                )
                loramoe = loramoe.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                loramoe_layers.append(loramoe)

                # Create wrapper
                wrapper = LoRAMoEWrapper(module, loramoe)
                wrappers.append(wrapper)

                # Replace module
                name_parts = name.split(".")
                parent = model
                for part in name_parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name_parts[-1], wrapper)

                break

    return LoRAMoEModel(
        backbone=model,
        loramoe_layers=loramoe_layers,
        wrappers=wrappers,
        balance_loss_coef=balance_loss_coef,
    )
