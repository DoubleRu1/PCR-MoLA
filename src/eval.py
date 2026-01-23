"""
Evaluation script for PCR-MoLA and baselines.

Supports:
- ICL (B1): In-context learning evaluation
- LoRA (B2): Standard LoRA evaluation
- LoRAMoE (B3): Token-conditioned MoE evaluation
- PCR-MoLA (Ours): Entity pair-conditioned routing evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import REDataset, REDataCollator, load_re_dataset
from .data.preprocess import (
    load_label_config,
    preprocess_dataset,
    create_synthetic_data,
)
from .models import create_pcr_mola_from_backbone, create_loramoe_model
from .models.baselines import ICLModel, select_exemplars, create_icl_prompt
from .models.pcr_mola import PCRMoLAConfig
from .utils import (
    set_seed,
    setup_logger,
    load_config,
    save_json,
    save_predictions,
    compute_metrics,
    normalize_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PCR-MoLA or baselines")

    # Config files
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to data config file",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )

    # Overrides
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--synthetic_samples", type=int, default=50)

    return parser.parse_args()


def load_model_for_eval(
    method: str,
    model_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
):
    """Load model for evaluation."""
    model_path = model_config["model"]["pretrained_model_name_or_path"]
    load_cfg = model_config["model"].get("load", {})

    # Determine dtype
    dtype_str = load_cfg.get("torch_dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=load_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load backbone
    backbone = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=load_cfg.get("trust_remote_code", True),
    )

    architecture = model_config["model"].get("architecture", "llama")

    if method == "icl":
        # ICL doesn't need model modification
        model = backbone
    elif method == "lora":
        # Load PEFT model
        from peft import PeftModel
        if checkpoint_path:
            model = PeftModel.from_pretrained(backbone, checkpoint_path)
        else:
            # Just use backbone for testing
            model = backbone
    elif method == "loramoe":
        loramoe_config = exp_config.get("loramoe", {})
        model = create_loramoe_model(
            model=backbone,
            num_experts=loramoe_config.get("num_experts", 4),
            rank=loramoe_config.get("rank", 8),
            alpha=loramoe_config.get("alpha", 32.0),
            top_k=loramoe_config.get("top_k", 2),
            dropout=0.0,  # No dropout during eval
            target_modules=loramoe_config.get("target_modules", ["q_proj", "v_proj"]),
            balance_loss_coef=0.0,
            freeze_backbone=True,
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    elif method == "pcr_mola":
        pcr_config = exp_config.get("pcr_mola", {})
        model = create_pcr_mola_from_backbone(
            model=backbone,
            architecture=architecture,
            router_dim=pcr_config.get("router_dim", 256),
            rank=pcr_config.get("rank", 8),
            alpha_scale=pcr_config.get("alpha_scale", 32.0),
            top_k=pcr_config.get("top_k", 2),
            balance_loss_coef=0.0,
            n_experts_low=pcr_config.get("n_experts_low", 2),
            n_experts_mid=pcr_config.get("n_experts_mid", 4),
            n_experts_high=pcr_config.get("n_experts_high", 8),
            pair_features=pcr_config.get("pair_features", "full"),
            router_dropout=0.0,  # No dropout during eval
            freeze_backbone=True,
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_icl(
    model,
    tokenizer,
    eval_examples: List[Dict[str, Any]],
    train_examples: List[Dict[str, Any]],
    label_config: Dict[str, Any],
    task_description: str,
    icl_config: Dict[str, Any],
    batch_size: int = 1,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """Evaluate using in-context learning."""
    label_list = label_config["label_list"]

    # Select exemplars
    exemplars = select_exemplars(
        examples=train_examples,
        num_exemplars=icl_config.get("num_exemplars", 5),
        strategy=icl_config.get("exemplar_selection", "stratified"),
        seed=icl_config.get("exemplar_seed", 42),
    )

    # Create ICL model wrapper
    icl_model = ICLModel(
        model=model,
        tokenizer=tokenizer,
        exemplars=exemplars,
        label_list=label_list,
        task_description=task_description,
        use_cot=icl_config.get("use_cot", False),
        max_new_tokens=max_new_tokens,
    )

    predictions = []
    references = []

    for example in tqdm(eval_examples, desc="ICL Evaluation"):
        pred = icl_model.predict(example["sentence"])
        predictions.append(pred)
        references.append(example["label"])

    # Compute metrics
    metrics = compute_metrics(
        predictions=predictions,
        references=references,
        label_set=label_list,
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "exemplars": [{"sentence": e["sentence"], "label": e["label"]} for e in exemplars],
    }


@torch.no_grad()
def evaluate_trained_model(
    model,
    tokenizer,
    eval_dataset: REDataset,
    collator: REDataCollator,
    method: str,
    label_list: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """Evaluate a trained model (LoRA, LoRAMoE, PCR-MoLA)."""
    from torch.utils.data import DataLoader

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    predictions = []
    references = []
    example_ids = []

    for batch in tqdm(eval_loader, desc=f"{method} Evaluation"):
        # Move batch to device
        input_ids = batch["input_ids"].to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
        attention_mask = batch["attention_mask"].to(input_ids.device)

        # Set entity info for PCR-MoLA
        if method == "pcr_mola" and hasattr(model, "set_entity_info"):
            model.set_entity_info(
                batch["e1_indices"].to(input_ids.device),
                batch["e2_indices"].to(input_ids.device),
                batch["e1_mask"].to(input_ids.device),
                batch["e2_mask"].to(input_ids.device),
                attention_mask,
            )

        # Generate
        if hasattr(model, "generate"):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Clear entity info
        if method == "pcr_mola" and hasattr(model, "clear_entity_info"):
            model.clear_entity_info()

        # Decode predictions
        for i in range(outputs.shape[0]):
            generated = tokenizer.decode(
                outputs[i][input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            predictions.append(generated)
            references.append(batch["label_texts"][i])
            example_ids.append(batch["example_ids"][i])

    # Compute metrics
    metrics = compute_metrics(
        predictions=predictions,
        references=references,
        label_set=label_list,
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "example_ids": example_ids,
    }


def main():
    args = parse_args()

    # Load configs
    from omegaconf import OmegaConf
    default_config = load_config("configs/default.yaml")
    exp_config = load_config(args.config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)

    config = OmegaConf.merge(default_config, exp_config, model_config, data_config)

    # Setup output directory
    output_dir = Path(args.output_dir or config.output.dir) / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name="eval",
        log_file=output_dir / "eval.log",
    )

    # Set seed
    set_seed(args.seed)

    # Get method
    method = config.experiment.method
    logger.info(f"Evaluation method: {method}")

    # Load label config
    label_config = load_label_config(config.dataset.labels_file)
    label_list = label_config["label_list"]

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_for_eval(
        method=method,
        model_config={"model": dict(config.model)},
        exp_config=dict(config),
        checkpoint_path=args.checkpoint,
    )

    # Load data
    logger.info(f"Loading {args.split} data...")

    if method == "icl":
        # For ICL, we need both train (for exemplars) and eval data
        if args.use_synthetic:
            train_examples = create_synthetic_data(
                config.dataset.name,
                args.synthetic_samples,
            )
            eval_examples = create_synthetic_data(
                config.dataset.name,
                args.synthetic_samples // 5,
            )
        else:
            train_examples = preprocess_dataset(
                config.dataset.name,
                config.dataset.data_dir,
                config.dataset.labels_file,
                dict(config.dataset.entity_markers),
                "train",
            )
            eval_examples = preprocess_dataset(
                config.dataset.name,
                config.dataset.data_dir,
                config.dataset.labels_file,
                dict(config.dataset.entity_markers),
                args.split,
            )

        # Evaluate ICL
        results = evaluate_icl(
            model=model,
            tokenizer=tokenizer,
            eval_examples=eval_examples,
            train_examples=train_examples,
            label_config=label_config,
            task_description=config.dataset.task_description,
            icl_config=dict(config.get("icl", {})),
            batch_size=1,  # ICL typically batch_size=1
            max_new_tokens=config.data.max_new_tokens,
        )

        # Save exemplars
        save_json(results["exemplars"], output_dir / "icl_exemplars.json")

    else:
        # For trained models
        eval_dataset = load_re_dataset(
            dataset_name=config.dataset.name,
            data_dir=config.dataset.data_dir,
            labels_file=config.dataset.labels_file,
            tokenizer=tokenizer,
            task_description=config.dataset.task_description,
            entity_markers=dict(config.dataset.entity_markers),
            max_length=config.data.max_length,
            split=args.split,
            use_synthetic=args.use_synthetic,
            synthetic_samples=args.synthetic_samples,
        )

        collator = REDataCollator(tokenizer=tokenizer)

        results = evaluate_trained_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            collator=collator,
            method=method,
            label_list=label_list,
            batch_size=args.batch_size,
            max_new_tokens=config.data.max_new_tokens,
        )

    # Log results
    metrics = results["metrics"]
    logger.info(f"Evaluation Results on {args.split}:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"  Invalid predictions: {metrics['invalid_count']}/{metrics['total_count']}")

    # Save results
    save_json(metrics, output_dir / "metrics.json")

    # Save predictions
    prediction_records = []
    for i, (pred, ref) in enumerate(zip(results["predictions"], results["references"])):
        record = {
            "id": results.get("example_ids", [i])[i] if i < len(results.get("example_ids", [])) else i,
            "prediction": pred,
            "reference": ref,
            "normalized_pred": normalize_label(pred, label_list),
            "correct": normalize_label(pred, label_list).lower() == ref.lower(),
        }
        prediction_records.append(record)

    save_predictions(prediction_records, output_dir / "predictions.jsonl")

    logger.info(f"Results saved to {output_dir}")

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint or 'None'}")
    print("-" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
