# PCR-MoLA Implementation Details

This document describes the key implementation details of PCR-MoLA, including the mathematical formulations and their code counterparts.

## Overview

PCR-MoLA (Entity Pair-Conditioned Routing with Layer-wise Adaptive Expert Allocation) is a LoRA-MoE adapter for relation extraction. Key innovations:

1. **Entity Pair-Conditioned Routing**: Routes based on entity pair representation, not individual tokens
2. **Layer-wise Adaptive Allocation**: Different number of experts per layer based on depth
3. **4-Feature Pair Representation**: Comprehensive entity pair encoding

## Mathematical Formulation

### 1. Entity Pair Representation

At each layer $l$, we compute entity representations by mean-pooling hidden states:

$$h_{e1}^{(l)} = \frac{1}{|I_{e1}|} \sum_{i \in I_{e1}} h_i^{(l)}$$

$$h_{e2}^{(l)} = \frac{1}{|I_{e2}|} \sum_{i \in I_{e2}} h_i^{(l)}$$

where $I_{e1}, I_{e2}$ are the token indices for each entity.

The raw pair feature is:

$$p_{raw}^{(l)} = [h_{e1}; h_{e2}; h_{e1} \odot h_{e2}; h_{e1} - h_{e2}] \in \mathbb{R}^{4d}$$

Projected to router space:

$$p^{(l)} = \text{ReLU}(W_p \cdot p_{raw}^{(l)} + b_p) \in \mathbb{R}^{d_p}$$

**Code location**: `src/models/pcr_mola.py:PairEncoder`

```python
def forward(self, hidden_states, e1_indices, e2_indices, e1_mask, e2_mask, attention_mask):
    # Pool entity hidden states
    h_e1 = self._pool_entity(hidden_states, e1_indices, e1_mask)
    h_e2 = self._pool_entity(hidden_states, e2_indices, e2_mask)

    # Full features: [h1; h2; h1*h2; h1-h2]
    p_raw = torch.cat([h_e1, h_e2, h_e1 * h_e2, h_e1 - h_e2], dim=-1)

    # Project with ReLU
    p = F.relu(self.projection(p_raw))
    return self.dropout(p)
```

### 2. Router and Top-K Selection

Router logits:

$$g^{(l)} = W_r^{(l)} \cdot p^{(l)} \in \mathbb{R}^{N^{(l)}}$$

Softmax probabilities:

$$\alpha^{(l)} = \text{softmax}(g^{(l)})$$

Top-K selection with renormalization:

$$S^{(l)} = \text{TopK}(\alpha^{(l)}, K)$$

$$\tilde{\alpha}_k^{(l)} = \frac{\alpha_k^{(l)}}{\sum_{j \in S^{(l)}} \alpha_j^{(l)}} \quad \text{for } k \in S^{(l)}$$

**Code location**: `src/models/pcr_mola.py:PCRMoLALayer.forward`

```python
# Router logits and softmax
router_logits = self.router(pair_repr)  # (B, num_experts)
router_probs = F.softmax(router_logits, dim=-1)

# Top-K selection
top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

# Renormalize
top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
```

### 3. Expert Computation

Each expert is a LoRA-style low-rank transformation:

$$E_k^{(l)}(x) = \frac{\alpha_{scale}}{r} \cdot W_{up,k}^{(l)} \cdot W_{down,k}^{(l)} \cdot x$$

where:
- $W_{down,k}^{(l)} \in \mathbb{R}^{r \times d}$: Down projection (Gaussian init)
- $W_{up,k}^{(l)} \in \mathbb{R}^{d \times r}$: Up projection (Zero init)
- $r$: LoRA rank
- $\alpha_{scale}$: Scaling factor (typically 32)

Aggregated expert output:

$$\Delta x^{(l)} = \sum_{k \in S^{(l)}} \tilde{\alpha}_k^{(l)} \cdot E_k^{(l)}(x)$$

**Code location**: `src/models/pcr_mola.py:PCRMoLALayer.forward`

```python
# Efficient batched computation using einsum
# x: (B, T, d), selected_down: (B, K, r, d), selected_up: (B, K, d, r)

# z = x @ W_down^T -> (B, T, K, r)
z = einsum(hidden_states, selected_down, "b t d, b k r d -> b t k r")

# y = z @ W_up^T -> (B, T, K, d)
y = einsum(z, selected_up, "b t k r, b k d r -> b t k d")

# Weighted sum: delta = sum_k (weight_k * y_k)
weights_expanded = top_k_weights.view(batch_size, 1, self.top_k, 1)
delta = (y * weights_expanded).sum(dim=2) * self.scaling
```

### 4. FFN Integration

The PCR-MoLA delta is added to the original FFN output:

$$x'^{(l)} = \text{FFN}^{(l)}(x) + \Delta x^{(l)}$$

**Code location**: `src/models/inject.py:PCRMoLAWrapper`

```python
def forward(self, hidden_states, *args, **kwargs):
    # Original FFN forward
    ffn_out = self.original_ffn(hidden_states, *args, **kwargs)

    # Add PCR-MoLA delta
    if self._entity_info is not None:
        delta = self.pcr_mola_layer(
            hidden_states,
            self._entity_info["e1_indices"],
            ...
        )
        ffn_out = ffn_out + delta

    return ffn_out
```

### 5. Balance Loss

Per-layer balance loss encourages even expert utilization:

$$f_k^{(l)} = \frac{1}{B} \sum_b \mathbf{1}[k \in S_b^{(l)}]$$

$$\bar{\alpha}_k^{(l)} = \frac{1}{B} \sum_b \alpha_{b,k}^{(l)}$$

$$L_{bal}^{(l)} = N^{(l)} \cdot \sum_k f_k^{(l)} \cdot \bar{\alpha}_k^{(l)}$$

Total balance loss:

$$L_{bal} = \frac{1}{L} \sum_l L_{bal}^{(l)}$$

Total loss:

$$L = L_{LM} + \lambda \cdot L_{bal}$$

**Code location**: `src/models/pcr_mola.py:PCRMoLAModel.compute_balance_loss`

```python
def compute_balance_loss(self):
    total_loss = 0.0
    for layer in self.layers:
        if layer.balance_stats is not None:
            stats = layer.balance_stats
            f_k = stats["f_k"]  # Expert selection frequency
            alpha_mean = stats["alpha_mean"]  # Mean router probability
            num_experts = stats["num_experts"]

            # L_bal^(l) = N * sum_k(f_k * alpha_mean_k)
            layer_loss = num_experts * (f_k * alpha_mean).sum()
            total_loss += layer_loss

    return total_loss / len(self.layers)
```

### 6. Layer-wise Expert Allocation

Experts are distributed based on layer depth:

- Layers $[0, L/3)$: $N_{low}$ experts (default: 2)
- Layers $[L/3, 2L/3)$: $N_{mid}$ experts (default: 4)
- Layers $[2L/3, L)$: $N_{high}$ experts (default: 8)

**Code location**: `src/models/pcr_mola.py:PCRMoLAConfig.get_num_experts`

```python
def get_num_experts(self, layer_idx: int) -> int:
    l1 = self.num_layers // 3
    l2 = 2 * self.num_layers // 3

    if layer_idx < l1:
        return self.n_experts_low
    elif layer_idx < l2:
        return self.n_experts_mid
    else:
        return self.n_experts_high
```

## Entity Token Index Extraction

Critical for routing: we must reliably identify entity token positions.

**Approach**: Insert special markers `<e1>`, `</e1>`, `<e2>`, `</e2>` and locate them after tokenization.

**Code location**: `src/data/datasets.py:find_entity_token_spans`

```python
def find_entity_token_spans(input_ids, tokenizer, e1_start_marker, ...):
    # Tokenize markers
    e1_start_ids = tokenizer.encode(e1_start_marker, add_special_tokens=False)
    e1_end_ids = tokenizer.encode(e1_end_marker, add_special_tokens=False)

    # Find marker positions in input_ids
    e1_start_pos = find_sublist(input_ids, e1_start_ids)
    e1_end_pos = find_sublist(input_ids, e1_end_ids, e1_start_pos + len(e1_start_ids))

    # Entity tokens are between markers
    e1_indices = list(range(e1_start_pos + len(e1_start_ids), e1_end_pos))

    return e1_indices, e2_indices
```

## Efficient Implementation Notes

### Batched Expert Computation

Instead of looping over experts, we use batched operations:

1. **Gather selected expert weights** using `torch.gather`
2. **Compute all expert outputs simultaneously** using `einsum`
3. **Weighted sum** using broadcasting

This is crucial for GPU efficiency.

### Memory Optimization

- **Gradient checkpointing**: Enabled by default for backbone
- **Mixed precision (bf16)**: Reduces memory by ~50%
- **Frozen backbone**: Only PCR-MoLA parameters are trainable

### Top-K Efficiency

Top-K routing means each sample only uses $K$ experts (typically 2), not all $N$ experts. This:
- Reduces computation by factor of $N/K$
- Enables larger expert counts without proportional cost increase

## Ablation Study Implementation

### Feature Ablation

Controlled via `pair_features` config:

| Mode | Features |
|------|----------|
| `full` | $[h_1; h_2; h_1 \odot h_2; h_1 - h_2]$ |
| `no_diff` | $[h_1; h_2; h_1 \odot h_2]$ |
| `no_interaction` | $[h_1; h_2; h_1 - h_2]$ |
| `context_only` | $h_{last}$ (sentence representation) |

### Routing Ablation

- **Static LoRA**: Uses standard LoRA (no routing)
- **Sentence-conditioned**: Uses `context_only` mode
- **No routing**: All experts weighted equally ($K = N$)

### Layer Allocation Ablation

- **Uniform**: Same $N$ for all layers
- **Reverse**: High $\rightarrow$ Mid $\rightarrow$ Low (opposite of ours)
- **High-only**: Experts only in top layers
- **Low-only**: Experts only in bottom layers

## Baseline Implementations

### B1: In-Context Learning

No training. Uses few-shot exemplars in prompt.

**Code location**: `src/models/baselines.py:ICLModel`

### B2: Standard LoRA

Uses PEFT library for static LoRA on attention projections.

**Code location**: `src/models/baselines.py:create_lora_model`

### B3: LoRAMoE

Token-conditioned routing (each token routes independently).

**Code location**: `src/models/baselines.py:LoRAMoELayer`

Key difference from PCR-MoLA:
- LoRAMoE: Routes per token based on token hidden state
- PCR-MoLA: Routes per sample based on entity pair representation (same for all tokens)
