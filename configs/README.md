# Configuration Guide

This document describes all configuration options for PCR-MoLA.

## Configuration Files

The system uses YAML configuration files organized hierarchically:

```
configs/
├── default.yaml          # Default values for all settings
├── data/                 # Dataset-specific configs
│   ├── chemprot.yaml
│   ├── ddi.yaml
│   └── gad.yaml
├── model/                # Model-specific configs
│   ├── qwen2.5-3b.yaml
│   ├── qwen2.5-7b.yaml
│   ├── qwen3-8b.yaml
│   └── llama3.1-8b.yaml
└── exp/                  # Experiment configs
    ├── baseline_b1.yaml  # ICL
    ├── baseline_b2.yaml  # Standard LoRA
    ├── baseline_b3.yaml  # LoRAMoE
    ├── ours.yaml         # PCR-MoLA
    └── ablate_*.yaml     # Ablation studies
```

## Configuration Hierarchy

Configs are merged in order (later overrides earlier):
1. `default.yaml`
2. Experiment config (e.g., `ours.yaml`)
3. Model config (e.g., `qwen2.5-7b.yaml`)
4. Data config (e.g., `chemprot.yaml`)
5. Command line overrides

## Default Values (default.yaml)

These are the recommended default values based on the paper:

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 3 | Number of training epochs |
| `training.batch_size` | 8 | Training batch size |
| `training.eval_batch_size` | 16 | Evaluation batch size |
| `training.learning_rate` | 3e-5 | Learning rate |
| `training.weight_decay` | 0.01 | Weight decay |
| `training.warmup_ratio` | 0.03 | Warmup ratio |
| `training.max_grad_norm` | 1.0 | Gradient clipping |
| `training.eval_every_steps` | 100 | Evaluation frequency |
| `training.seed` | 42 | Random seed |
| `training.bf16` | true | Use bfloat16 |
| `training.gradient_checkpointing` | true | Enable gradient checkpointing |

### Data Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.max_length` | 128 | Maximum sequence length |
| `data.max_new_tokens` | 16 | Maximum tokens to generate |
| `data.num_workers` | 4 | DataLoader workers |

### PCR-MoLA Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pcr_mola.router_dim` | 256 | Router projection dimension |
| `pcr_mola.rank` | 8 | LoRA rank for experts |
| `pcr_mola.alpha_scale` | 32 | LoRA alpha scaling |
| `pcr_mola.top_k` | 2 | Number of experts to select |
| `pcr_mola.balance_loss_coef` | 0.01 | Balance loss coefficient (λ) |
| `pcr_mola.n_experts_low` | 2 | Experts for low layers [0, L/3) |
| `pcr_mola.n_experts_mid` | 4 | Experts for mid layers [L/3, 2L/3) |
| `pcr_mola.n_experts_high` | 8 | Experts for high layers [2L/3, L) |
| `pcr_mola.pair_features` | "full" | Feature mode (see below) |
| `pcr_mola.router_dropout` | 0.1 | Router dropout |

### Pair Features Mode

| Mode | Features | Dimension |
|------|----------|-----------|
| `"full"` | [h1; h2; h1⊙h2; h1-h2] | 4d |
| `"no_diff"` | [h1; h2; h1⊙h2] | 3d |
| `"no_interaction"` | [h1; h2; h1-h2] | 3d |
| `"context_only"` | h_last | d |

### LoRA Baseline Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.rank` | 8 | LoRA rank |
| `lora.alpha` | 32 | LoRA alpha |
| `lora.dropout` | 0.1 | LoRA dropout |
| `lora.target_modules` | ["q_proj", "v_proj"] | Modules to adapt |

### LoRAMoE Baseline Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loramoe.rank` | 8 | LoRA rank |
| `loramoe.alpha` | 32 | LoRA alpha |
| `loramoe.num_experts` | 4 | Number of experts |
| `loramoe.top_k` | 2 | Top-K routing |
| `loramoe.balance_loss_coef` | 0.01 | Balance loss coefficient |

### ICL Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `icl.num_exemplars` | 5 | Number of few-shot examples |
| `icl.exemplar_selection` | "stratified" | Selection strategy |
| `icl.use_cot` | false | Chain-of-thought |

## Model Configuration

Example model config (`qwen2.5-7b.yaml`):

```yaml
model:
  name: "qwen2.5-7b"
  pretrained_model_name_or_path: "Qwen/Qwen2.5-7B"
  architecture: "qwen2"
  hidden_size: 3584
  num_hidden_layers: 28

  # FFN module pattern for injection
  ffn_module_pattern: "mlp"

  # Attention projections for LoRA
  attention_projections:
    q_proj: "q_proj"
    v_proj: "v_proj"

  tokenizer:
    use_fast: true
    padding_side: "left"

  load:
    torch_dtype: "bfloat16"
    device_map: "auto"
    trust_remote_code: true
```

## Dataset Configuration

Example data config (`chemprot.yaml`):

```yaml
dataset:
  name: "chemprot"
  data_dir: "data/chemprot"
  labels_file: "data/labels/chemprot.json"

  entity_markers:
    e1_start: "<e1>"
    e1_end: "</e1>"
    e2_start: "<e2>"
    e2_end: "</e2>"

  task_description: "Classify the chemical-protein interaction relation."
```

## Experiment Configuration

Example experiment config (`ours.yaml`):

```yaml
experiment:
  name: "pcr_mola"
  method: "pcr_mola"
  description: "PCR-MoLA with entity pair-conditioned routing"

pcr_mola:
  router_dim: 256
  rank: 8
  alpha_scale: 32
  top_k: 2
  balance_loss_coef: 0.01
  n_experts_low: 2
  n_experts_mid: 4
  n_experts_high: 8
  pair_features: "full"

training:
  do_train: true
  do_eval: true

output:
  subdir: "pcr_mola"
```

## Sweep Configuration

For ablation studies, use sweep configs:

```yaml
sweep:
  parameter: "pair_features"
  values:
    - name: "context_only"
      method: "pcr_mola"
      pcr_mola:
        pair_features: "context_only"

    - name: "full_features"
      method: "pcr_mola"
      pcr_mola:
        pair_features: "full"
```

## Command Line Overrides

Override any config value from command line:

```bash
python -m src.train \
    --config configs/exp/ours.yaml \
    --model_config configs/model/qwen2.5-7b.yaml \
    --data_config configs/data/chemprot.yaml \
    --seed 123 \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-5
```

## Memory Optimization Tips

For limited GPU memory:

1. **Reduce batch size**: `--batch_size 2`
2. **Enable gradient accumulation**:
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 4  # Effective batch size = 8
   ```
3. **Use smaller model**: Start with `qwen2.5-3b.yaml`
4. **Reduce expert count**:
   ```yaml
   pcr_mola:
     n_experts_low: 1
     n_experts_mid: 2
     n_experts_high: 4
   ```
