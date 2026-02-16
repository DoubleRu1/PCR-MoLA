#!/bin/bash
# Local experiment runner for PCR-MoLA
# Run all experiments on local machine (CPU/MPS)

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
USE_SYNTHETIC="${USE_SYNTHETIC:-false}"
SKIP_BASELINES="${SKIP_BASELINES:-false}"
SKIP_ABLATIONS="${SKIP_ABLATIONS:-true}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Model configs
MODELS=(
    "configs/model/qwen2.5-3b.yaml"
    # Add more models if needed:
    # "configs/model/qwen2.5-7b.yaml"
)

# Data configs (only chemprot available)
DATASETS=(
    "configs/data/chemprot.yaml"
)

# Synthetic data flag
SYNTHETIC_FLAG=""
if [ "$USE_SYNTHETIC" = "true" ]; then
    SYNTHETIC_FLAG="--use_synthetic --synthetic_samples 200"
    echo "Using synthetic data for testing"
fi

echo "=========================================="
echo "PCR-MoLA Local Experiment Runner"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

# Create output directories
mkdir -p "$OUTPUT_DIR/tables"
mkdir -p "$OUTPUT_DIR/logs"

# Use CPU explicitly
export CUDA_VISIBLE_DEVICES=""
export MPS_VISIBLE_DEVICES=""

PYTHON="${HOME}/Code/PCR-MoLA/.venv/bin/python"

# ==========================================
# Main Comparison Experiments (B1, B2, B3, Ours)
# ==========================================

if [ "$SKIP_BASELINES" != "true" ]; then
    echo ""
    echo "=========================================="
    echo "Running Main Comparison Experiments"
    echo "=========================================="

    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$MODEL" .yaml)

        for DATA in "${DATASETS[@]}"; do
            DATA_NAME=$(basename "$DATA" .yaml)

            # B1: ICL (no training needed)
            echo ""
            echo "--- B1 ICL: $MODEL_NAME on $DATA_NAME ---"
            $PYTHON -m src.eval \
                --config configs/exp/baseline_b1.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/b1_icl_${MODEL_NAME}_${DATA_NAME}" \
                --split test \
                $SYNTHETIC_FLAG \
                2>&1 | tee "$OUTPUT_DIR/logs/b1_icl_${MODEL_NAME}_${DATA_NAME}.log"

            # B2: Standard LoRA
            echo ""
            echo "--- B2 LoRA: $MODEL_NAME on $DATA_NAME ---"
            $PYTHON -m src.train \
                --config configs/exp/baseline_b2.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/b2_lora_${MODEL_NAME}_${DATA_NAME}" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                $SYNTHETIC_FLAG \
                2>&1 | tee "$OUTPUT_DIR/logs/b2_lora_${MODEL_NAME}_${DATA_NAME}.log"

            # B3: LoRAMoE
            echo ""
            echo "--- B3 LoRAMoE: $MODEL_NAME on $DATA_NAME ---"
            $PYTHON -m src.train \
                --config configs/exp/baseline_b3.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/b3_loramoe_${MODEL_NAME}_${DATA_NAME}" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                $SYNTHETIC_FLAG \
                2>&1 | tee "$OUTPUT_DIR/logs/b3_loramoe_${MODEL_NAME}_${DATA_NAME}.log"

            # Ours: PCR-MoLA
            echo ""
            echo "--- PCR-MoLA: $MODEL_NAME on $DATA_NAME ---"
            $PYTHON -m src.train \
                --config configs/exp/ours.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/pcr_mola_${MODEL_NAME}_${DATA_NAME}" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                $SYNTHETIC_FLAG \
                2>&1 | tee "$OUTPUT_DIR/logs/pcr_mola_${MODEL_NAME}_${DATA_NAME}.log"

        done
    done
fi

# ==========================================
# Ablation Studies
# ==========================================

if [ "$SKIP_ABLATIONS" != "true" ]; then
    echo ""
    echo "=========================================="
    echo "Running Ablation Studies"
    echo "=========================================="

    # Use first model for ablations
    ABLATION_MODEL="${MODELS[0]}"
    ABLATION_MODEL_NAME=$(basename "$ABLATION_MODEL" .yaml)

    # Ablation 1: Routing Condition
    echo ""
    echo "--- Ablation: Routing Condition ---"
    $PYTHON -m src.run_sweep \
        --sweep_config configs/exp/ablate_routing.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_routing" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_routing.log"

    # Ablation 2: Entity Pair Features
    echo ""
    echo "--- Ablation: Entity Pair Features ---"
    $PYTHON -m src.run_sweep \
        --sweep_config configs/exp/ablate_features.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_features" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_features.log"

    # Ablation 3: Expert Numbers and Balance Loss
    echo ""
    echo "--- Ablation: Expert Numbers & Balance Loss ---"
    $PYTHON -m src.run_sweep \
        --sweep_config configs/exp/ablate_experts_balance.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_experts_balance" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_experts_balance.log"

    # Ablation 4: Layer-wise Allocation
    echo ""
    echo "--- Ablation: Layer-wise Allocation ---"
    $PYTHON -m src.run_sweep \
        --sweep_config configs/exp/ablate_layerwise_alloc.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_layerwise_alloc" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_layerwise_alloc.log"
fi

# ==========================================
# Generate Summary Tables
# ==========================================

echo ""
echo "=========================================="
echo "Generating Summary Tables"
echo "=========================================="

$PYTHON -m src.generate_tables \
    --output_dir "$OUTPUT_DIR" \
    --tables_dir "$OUTPUT_DIR/tables"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Tables saved to: $OUTPUT_DIR/tables"
echo "=========================================="
