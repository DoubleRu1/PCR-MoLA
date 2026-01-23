# PCR-MoLA: Entity Pair-Conditioned Routing with Layer-wise Adaptive Expert Allocation

This repository implements PCR-MoLA, a LoRA-MoE adapter for relation extraction that uses entity pair-conditioned routing and layer-wise adaptive expert allocation.

## Features

- **PCR-MoLA**: Entity pair-conditioned routing with 4-feature representation
- **Layer-wise Allocation**: Adaptive expert distribution across transformer layers
- **Multiple Backbones**: Support for Qwen-2.5 (3B, 7B), Qwen-3 (8B), Llama-3.1 (8B)
- **Baselines**: ICL, Standard LoRA, LoRAMoE
- **Datasets**: ChemProt, DDI, GAD
- **Metrics**: Accuracy, Macro-F1, Weighted-F1

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended: 24GB+ VRAM for 7B+ models)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup with uv

```bash
# Clone the repository
git clone <repo-url>
cd pcr-mola

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# For optional dependencies (DeepSpeed, Flash Attention)
uv sync --extra deepspeed
uv sync --extra flash-attn
```

### Manual Installation (alternative)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Project Structure

```
pcr-mola/
├── configs/
│   ├── default.yaml           # Default hyperparameters
│   ├── data/                   # Dataset configs
│   │   ├── chemprot.yaml
│   │   ├── ddi.yaml
│   │   └── gad.yaml
│   ├── model/                  # Model configs
│   │   ├── qwen2.5-3b.yaml
│   │   ├── qwen2.5-7b.yaml
│   │   ├── qwen3-8b.yaml
│   │   └── llama3.1-8b.yaml
│   └── exp/                    # Experiment configs
│       ├── baseline_b1.yaml    # ICL
│       ├── baseline_b2.yaml    # Standard LoRA
│       ├── baseline_b3.yaml    # LoRAMoE
│       ├── ours.yaml           # PCR-MoLA
│       └── ablate_*.yaml       # Ablation studies
├── data/
│   └── labels/                 # Label configurations
├── src/
│   ├── data/                   # Data processing
│   ├── models/                 # Model implementations
│   │   ├── pcr_mola.py        # Core PCR-MoLA
│   │   ├── inject.py          # FFN injection
│   │   └── baselines.py       # Baseline methods
│   ├── utils/                  # Utilities
│   ├── train.py               # Training script
│   ├── eval.py                # Evaluation script
│   ├── run_sweep.py           # Sweep runner
│   └── generate_tables.py     # Table generation
├── scripts/
│   └── run_all.sh             # Run all experiments
├── outputs/                    # Results (auto-generated)
└── docs/
    └── implementation.md      # Implementation details
```

## Data Preparation

### Download Datasets

Place dataset files in the corresponding directories:

```
data/
├── chemprot/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── ddi/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── gad/
    ├── train.json
    ├── dev.json
    └── test.json
```

### Data Format

Each JSON file should contain a list of examples:

```json
[
  {
    "id": "0",
    "sentence": "The drug aspirin interacts with protein COX-2.",
    "e1_start": 9,
    "e1_end": 16,
    "e2_start": 37,
    "e2_end": 42,
    "label": "inhibitor"
  }
]
```

### Using Synthetic Data (for testing)

```bash
python -m src.train ... --use_synthetic --synthetic_samples 200
```

## Quick Start

### Training PCR-MoLA

```bash
# Activate environment
source .venv/bin/activate

# Train PCR-MoLA on ChemProt with Qwen2.5-3B
python -m src.train \
    --config configs/exp/ours.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml \
    --output_dir outputs/pcr_mola_qwen2.5-3b_chemprot

# With synthetic data (for testing)
python -m src.train \
    --config configs/exp/ours.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml \
    --output_dir outputs/test_run \
    --use_synthetic --synthetic_samples 200
```

### Training Baselines

```bash
# B2: Standard LoRA
python -m src.train \
    --config configs/exp/baseline_b2.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml

# B3: LoRAMoE
python -m src.train \
    --config configs/exp/baseline_b3.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml
```

### Evaluation

```bash
# Evaluate trained model
python -m src.eval \
    --config configs/exp/ours.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml \
    --checkpoint outputs/pcr_mola_qwen2.5-3b_chemprot/best_model.pt \
    --split test

# B1: ICL evaluation (no training needed)
python -m src.eval \
    --config configs/exp/baseline_b1.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml \
    --split test
```

### Run All Experiments

```bash
# Run all experiments (main comparison + ablations)
bash scripts/run_all.sh

# With synthetic data (for testing pipeline)
USE_SYNTHETIC=true bash scripts/run_all.sh

# Skip baselines, only run ablations
SKIP_BASELINES=true bash scripts/run_all.sh
```

### Generate Summary Tables

```bash
python -m src.generate_tables --output_dir outputs --tables_dir outputs/tables
```

## Configuration

### Default Hyperparameters (configs/default.yaml)

```yaml
training:
  epochs: 3
  batch_size: 8
  learning_rate: 3.0e-5
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true

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
```

### Command Line Overrides

```bash
python -m src.train \
    --config configs/exp/ours.yaml \
    --model_config configs/model/qwen2.5-3b.yaml \
    --data_config configs/data/chemprot.yaml \
    --seed 42 \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-5
```

## Adding New Backbones

1. Create a model config file in `configs/model/`:

```yaml
model:
  name: "your-model"
  pretrained_model_name_or_path: "your-org/your-model"
  architecture: "llama"  # or "qwen2"
  hidden_size: 4096
  num_hidden_layers: 32
  ffn_module_pattern: "mlp"
  attention_projections:
    q_proj: "q_proj"
    v_proj: "v_proj"
```

2. If the model has different FFN naming, update `FFN_PATTERNS` in `src/models/inject.py`.

## Output Tables

After running experiments, tables are saved to `outputs/tables/`:

| Table | Description |
|-------|-------------|
| `compare_main.csv/md` | Main comparison (B1/B2/B3/Ours × backbones × datasets) |
| `ablate_routing.csv/md` | Routing condition ablation |
| `ablate_features.csv/md` | Entity pair feature ablation |
| `ablate_experts_balance.csv/md` | Expert number & balance loss ablation |
| `ablate_layerwise_alloc.csv/md` | Layer-wise allocation ablation |

## Reproducibility

- All experiments use fixed random seed (default: 42)
- Git hash and full config are saved with each run
- Checkpoints include training state for resumption

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python -m src.train ... --batch_size 4

# Enable gradient accumulation in config
training:
  batch_size: 2
  gradient_accumulation_steps: 4
```

### Model Loading Issues

```bash
# Set trust_remote_code in model config
model:
  load:
    trust_remote_code: true
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Set specific GPU
CUDA_VISIBLE_DEVICES=0 python -m src.train ...
```

## Citation

```bibtex
@article{pcr-mola,
  title={PCR-MoLA: Entity Pair-Conditioned Routing with Layer-wise Adaptive Expert Allocation},
  author={...},
  year={2024}
}
```

## License

MIT License
