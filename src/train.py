"""
Training script for PCR-MoLA and baselines.

Supports:
- PCR-MoLA (ours)
- Standard LoRA (B2)
- LoRAMoE (B3)
- Accelerate for distributed training
- Mixed precision (bf16/fp16)
- Gradient checkpointing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import REDataset, REDataCollator, load_re_dataset
from .data.preprocess import load_label_config
from .models import (
    create_pcr_mola_from_backbone,
    create_lora_model,
    create_loramoe_model,
)
from .models.pcr_mola import PCRMoLAConfig
from .utils import (
    set_seed,
    setup_logger,
    log_config,
    log_trainable_params,
    save_json,
    load_config,
    save_experiment_info,
    compute_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCR-MoLA or baselines")

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

    # Overrides
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--synthetic_samples", type=int, default=200)

    return parser.parse_args()


def load_backbone(
    model_config: Dict[str, Any],
    use_gradient_checkpointing: bool = True,
) -> tuple:
    """Load backbone model and tokenizer."""
    model_path = model_config["model"]["pretrained_model_name_or_path"]
    load_config = model_config["model"].get("load", {})

    # Determine dtype
    dtype_str = load_config.get("torch_dtype", "bfloat16")
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=load_config.get("trust_remote_code", True),
        use_fast=model_config["model"].get("tokenizer", {}).get("use_fast", True),
    )

    # Set padding side
    padding_side = model_config["model"].get("tokenizer", {}).get("padding_side", "left")
    tokenizer.padding_side = padding_side

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=load_config.get("device_map", "auto"),
        trust_remote_code=load_config.get("trust_remote_code", True),
    )

    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def create_model(
    backbone: AutoModelForCausalLM,
    method: str,
    exp_config: Dict[str, Any],
    model_config: Dict[str, Any],
):
    """Create model based on method."""
    architecture = model_config["model"].get("architecture", "llama")

    if method == "pcr_mola":
        pcr_config = exp_config.get("pcr_mola", {})
        model = create_pcr_mola_from_backbone(
            model=backbone,
            architecture=architecture,
            router_dim=pcr_config.get("router_dim", 256),
            rank=pcr_config.get("rank", 8),
            alpha_scale=pcr_config.get("alpha_scale", 32.0),
            top_k=pcr_config.get("top_k", 2),
            balance_loss_coef=pcr_config.get("balance_loss_coef", 0.01),
            n_experts_low=pcr_config.get("n_experts_low", 2),
            n_experts_mid=pcr_config.get("n_experts_mid", 4),
            n_experts_high=pcr_config.get("n_experts_high", 8),
            pair_features=pcr_config.get("pair_features", "full"),
            router_dropout=pcr_config.get("router_dropout", 0.1),
            freeze_backbone=True,
        )
    elif method == "lora":
        lora_config = exp_config.get("lora", {})
        model = create_lora_model(
            model=backbone,
            rank=lora_config.get("rank", 8),
            alpha=lora_config.get("alpha", 32),
            dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        )
    elif method == "loramoe":
        loramoe_config = exp_config.get("loramoe", {})
        model = create_loramoe_model(
            model=backbone,
            num_experts=loramoe_config.get("num_experts", 4),
            rank=loramoe_config.get("rank", 8),
            alpha=loramoe_config.get("alpha", 32.0),
            top_k=loramoe_config.get("top_k", 2),
            dropout=loramoe_config.get("dropout", 0.1),
            target_modules=loramoe_config.get("target_modules", ["q_proj", "v_proj"]),
            balance_loss_coef=loramoe_config.get("balance_loss_coef", 0.01),
            freeze_backbone=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
    method: str,
    epoch: int,
    logger,
    logging_steps: int = 10,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_balance_loss = 0
    num_steps = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )

    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            # Prepare inputs based on method
            if method == "pcr_mola":
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    e1_indices=batch["e1_indices"],
                    e2_indices=batch["e2_indices"],
                    e1_mask=batch["e1_mask"],
                    e2_mask=batch["e2_mask"],
                )
                loss = outputs.total_loss if hasattr(outputs, "total_loss") else outputs.loss
                lm_loss = outputs.lm_loss if hasattr(outputs, "lm_loss") else outputs.loss
                balance_loss = outputs.balance_loss if hasattr(outputs, "balance_loss") else 0.0
            elif method == "loramoe":
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.total_loss if hasattr(outputs, "total_loss") else outputs.loss
                lm_loss = outputs.lm_loss if hasattr(outputs, "lm_loss") else outputs.loss
                balance_loss = outputs.balance_loss if hasattr(outputs, "balance_loss") else 0.0
            else:  # lora
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                lm_loss = loss
                balance_loss = 0.0

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_lm_loss += lm_loss.item() if torch.is_tensor(lm_loss) else lm_loss
        total_balance_loss += balance_loss.item() if torch.is_tensor(balance_loss) else balance_loss
        num_steps += 1

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{total_loss / num_steps:.4f}",
            "lm": f"{total_lm_loss / num_steps:.4f}",
            "bal": f"{total_balance_loss / num_steps:.4f}",
        })

        # Log periodically
        if step > 0 and step % logging_steps == 0 and accelerator.is_local_main_process:
            logger.info(
                f"Epoch {epoch} Step {step}: "
                f"loss={total_loss / num_steps:.4f} "
                f"lm_loss={total_lm_loss / num_steps:.4f} "
                f"balance_loss={total_balance_loss / num_steps:.4f}"
            )

    return {
        "loss": total_loss / num_steps,
        "lm_loss": total_lm_loss / num_steps,
        "balance_loss": total_balance_loss / num_steps,
    }


@torch.no_grad()
def evaluate(
    model,
    eval_loader: DataLoader,
    tokenizer,
    accelerator: Accelerator,
    method: str,
    label_list: list,
    max_new_tokens: int = 16,
):
    """Evaluate model on validation/test set."""
    model.eval()

    all_predictions = []
    all_references = []

    for batch in tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        # For evaluation, we generate predictions
        # Get input up to "### Answer:"
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Generate
        if method == "pcr_mola":
            # Need to set entity info for generation
            if hasattr(model, "set_entity_info"):
                model.set_entity_info(
                    batch["e1_indices"],
                    batch["e2_indices"],
                    batch["e1_mask"],
                    batch["e2_mask"],
                    attention_mask,
                )

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

            if hasattr(model, "clear_entity_info"):
                model.clear_entity_info()
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode generated tokens (only new tokens)
        for i in range(outputs.shape[0]):
            generated = tokenizer.decode(
                outputs[i][input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            all_predictions.append(generated)
            all_references.append(batch["label_texts"][i])

    # Gather from all processes
    all_predictions = accelerator.gather_for_metrics(all_predictions)
    all_references = accelerator.gather_for_metrics(all_references)

    # Compute metrics
    metrics = compute_metrics(
        predictions=all_predictions,
        references=all_references,
        label_set=label_list,
    )

    return metrics, all_predictions, all_references


def main():
    args = parse_args()

    # Load configs
    default_config = load_config("configs/default.yaml")
    exp_config = load_config(args.config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)

    # Merge configs (exp_config overrides default_config)
    from omegaconf import OmegaConf
    config = OmegaConf.merge(default_config, exp_config, model_config, data_config)

    # Apply command line overrides
    if args.output_dir:
        config.output.dir = args.output_dir
    if args.seed is not None:
        config.training.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate

    # Setup output directory
    output_dir = Path(config.output.dir) / config.output.get("subdir", "run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name="train",
        log_file=output_dir / "train.log",
    )

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision="bf16" if config.training.bf16 else ("fp16" if config.training.fp16 else "no"),
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    # Set seed
    seed = config.training.seed
    set_seed(seed)
    accelerate_set_seed(seed)

    # Log config
    if accelerator.is_local_main_process:
        log_config(logger, config)
        save_experiment_info(output_dir, config)

    # Get method
    method = config.experiment.method
    logger.info(f"Training method: {method}")

    # Skip training for ICL
    if method == "icl":
        logger.info("ICL method does not require training. Use eval.py instead.")
        return

    # Load backbone
    logger.info("Loading backbone model...")
    backbone, tokenizer = load_backbone(
        dict(config.model) if hasattr(config, "model") else {"model": config.model},
        use_gradient_checkpointing=config.training.gradient_checkpointing,
    )

    # Load label config
    label_config = load_label_config(config.dataset.labels_file)
    label_list = label_config["label_list"]

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = load_re_dataset(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        labels_file=config.dataset.labels_file,
        tokenizer=tokenizer,
        task_description=config.dataset.task_description,
        entity_markers=dict(config.dataset.entity_markers),
        max_length=config.data.max_length,
        split="train",
        use_synthetic=args.use_synthetic,
        synthetic_samples=args.synthetic_samples,
    )

    eval_dataset = load_re_dataset(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        labels_file=config.dataset.labels_file,
        tokenizer=tokenizer,
        task_description=config.dataset.task_description,
        entity_markers=dict(config.dataset.entity_markers),
        max_length=config.data.max_length,
        split="dev",
        use_synthetic=args.use_synthetic,
        synthetic_samples=args.synthetic_samples // 5,
    )

    # Create data collator
    collator = REDataCollator(tokenizer=tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.data.num_workers,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
    )

    # Create model
    logger.info(f"Creating {method} model...")
    model = create_model(
        backbone=backbone,
        method=method,
        exp_config=dict(config),
        model_config={"model": dict(config.model)},
    )

    # Log trainable parameters
    if accelerator.is_local_main_process:
        param_info = log_trainable_params(logger, model)
        save_json(param_info, output_dir / "param_info.json")

    # Create optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    num_training_steps = len(train_loader) * config.training.epochs
    num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Training loop
    logger.info("Starting training...")
    best_f1 = 0.0

    for epoch in range(1, config.training.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            method=method,
            epoch=epoch,
            logger=logger,
            logging_steps=config.training.logging_steps,
        )

        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch} training: {train_metrics}")

        # Evaluate
        eval_metrics, predictions, references = evaluate(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            accelerator=accelerator,
            method=method,
            label_list=label_list,
            max_new_tokens=config.data.max_new_tokens,
        )

        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch {epoch} eval: "
                f"acc={eval_metrics['accuracy']:.4f} "
                f"macro_f1={eval_metrics['macro_f1']:.4f} "
                f"weighted_f1={eval_metrics['weighted_f1']:.4f}"
            )

            # Save metrics
            save_json(
                {"epoch": epoch, "train": train_metrics, "eval": eval_metrics},
                output_dir / f"metrics_epoch_{epoch}.json",
            )

            # Save best model
            if eval_metrics["macro_f1"] > best_f1:
                best_f1 = eval_metrics["macro_f1"]
                logger.info(f"New best F1: {best_f1:.4f}")

                # Save model
                unwrapped_model = accelerator.unwrap_model(model)
                if method == "lora":
                    unwrapped_model.save_pretrained(output_dir / "best_model")
                else:
                    torch.save(
                        unwrapped_model.state_dict(),
                        output_dir / "best_model.pt",
                    )

                save_json(eval_metrics, output_dir / "best_metrics.json")

    logger.info(f"Training complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
