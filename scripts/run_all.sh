#!/bin/bash
# Run all experiments for PCR-MoLA paper reproduction
# This script runs all baselines, main comparisons, and ablation studies

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
USE_SYNTHETIC="${USE_SYNTHETIC:-false}"
SKIP_BASELINES="${SKIP_BASELINES:-false}"
SKIP_ABLATIONS="${SKIP_ABLATIONS:-false}"

# Model configs
MODELS=(
    "configs/model/qwen2.5-3b.yaml"
    # "configs/model/qwen2.5-7b.yaml"
    # "configs/model/qwen3-8b.yaml"
    # "configs/model/llama3.1-8b.yaml"
)

# Data configs
DATASETS=(
    "configs/data/chemprot.yaml"
    "configs/data/ddi.yaml"
    "configs/data/gad.yaml"
)

# Synthetic data flag
SYNTHETIC_FLAG=""
if [ "$USE_SYNTHETIC" = "true" ]; then
    SYNTHETIC_FLAG="--use_synthetic --synthetic_samples 200"
    echo "Using synthetic data for testing"
fi

echo "=========================================="
echo "PCR-MoLA Experiment Runner"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "=========================================="

# Create output directories
mkdir -p "$OUTPUT_DIR/tables"
mkdir -p "$OUTPUT_DIR/logs"

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
            python -m src.eval \
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
            python -m src.train \
                --config configs/exp/baseline_b2.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/b2_lora_${MODEL_NAME}_${DATA_NAME}" \
                $SYNTHETIC_FLAG \
                2>&1 | tee "$OUTPUT_DIR/logs/b2_lora_${MODEL_NAME}_${DATA_NAME}.log"

            # B3: LoRAMoE
            echo ""
            echo "--- B3 LoRAMoE: $MODEL_NAME on $DATA_NAME ---"
            python -m src.train \
                --config configs/exp/baseline_b3.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/b3_loramoe_${MODEL_NAME}_${DATA_NAME}" \
                $SYNTHETIC_FLAG \
                2>&1 | tee "$OUTPUT_DIR/logs/b3_loramoe_${MODEL_NAME}_${DATA_NAME}.log"

            # Ours: PCR-MoLA
            echo ""
            echo "--- PCR-MoLA: $MODEL_NAME on $DATA_NAME ---"
            python -m src.train \
                --config configs/exp/ours.yaml \
                --model_config "$MODEL" \
                --data_config "$DATA" \
                --output_dir "$OUTPUT_DIR/pcr_mola_${MODEL_NAME}_${DATA_NAME}" \
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
    python -m src.run_sweep \
        --sweep_config configs/exp/ablate_routing.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_routing" \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_routing.log"

    # Ablation 2: Entity Pair Features
    echo ""
    echo "--- Ablation: Entity Pair Features ---"
    python -m src.run_sweep \
        --sweep_config configs/exp/ablate_features.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_features" \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_features.log"

    # Ablation 3: Expert Numbers and Balance Loss
    echo ""
    echo "--- Ablation: Expert Numbers & Balance Loss ---"
    python -m src.run_sweep \
        --sweep_config configs/exp/ablate_experts_balance.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_experts_balance" \
        $SYNTHETIC_FLAG \
        2>&1 | tee "$OUTPUT_DIR/logs/ablate_experts_balance.log"

    # Ablation 4: Layer-wise Allocation
    echo ""
    echo "--- Ablation: Layer-wise Allocation ---"
    python -m src.run_sweep \
        --sweep_config configs/exp/ablate_layerwise_alloc.yaml \
        --model_configs "$ABLATION_MODEL" \
        --data_configs "${DATASETS[@]}" \
        --output_dir "$OUTPUT_DIR/ablate_layerwise_alloc" \
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

python -m src.generate_tables \
    --output_dir "$OUTPUT_DIR" \
    --tables_dir "$OUTPUT_DIR/tables"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Tables saved to: $OUTPUT_DIR/tables"
echo "=========================================="
