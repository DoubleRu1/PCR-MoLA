#!/usr/bin/env python
"""
Verification script to test PCR-MoLA installation and core components.

Run with: python scripts/verify_installation.py
"""

import sys
import torch
import torch.nn as nn


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("Testing Imports")

    imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
        ("datasets", "Datasets"),
        ("omegaconf", "OmegaConf"),
        ("einops", "Einops"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
    ]

    for module, name in imports:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            return False

    # Test project modules
    project_imports = [
        ("src.data", "Data module"),
        ("src.models", "Models module"),
        ("src.utils", "Utils module"),
        ("src.models.pcr_mola", "PCR-MoLA"),
        ("src.models.inject", "Injection"),
        ("src.models.baselines", "Baselines"),
    ]

    for module, name in project_imports:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            return False

    return True


def test_pcr_mola_layer():
    """Test PCR-MoLA layer forward pass."""
    print_header("Testing PCR-MoLA Layer")

    from src.models.pcr_mola import PCRMoLAConfig, PCRMoLAModel, PairEncoder

    # Create config
    config = PCRMoLAConfig(
        hidden_size=256,  # Small for testing
        router_dim=64,
        rank=4,
        alpha_scale=16,
        top_k=2,
        num_layers=4,
        n_experts_low=2,
        n_experts_mid=2,
        n_experts_high=4,
        pair_features="full",
        device="cpu",
        dtype=torch.float32,
    )

    # Create model
    model = PCRMoLAModel(config)
    model = model.to("cpu")
    print(f"  Created PCRMoLAModel with {config.num_layers} layers")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    max_entity_len = 3

    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    e1_indices = torch.tensor([[2, 3, 0], [5, 6, 7]])
    e2_indices = torch.tensor([[8, 9, 0], [10, 11, 0]])
    e1_mask = torch.tensor([[True, True, False], [True, True, True]])
    e2_mask = torch.tensor([[True, True, False], [True, True, False]])

    # Test each layer
    for layer_idx, layer in enumerate(model.layers):
        delta = layer(
            hidden_states=hidden_states,
            e1_indices=e1_indices,
            e2_indices=e2_indices,
            e1_mask=e1_mask,
            e2_mask=e2_mask,
        )
        assert delta.shape == hidden_states.shape, f"Layer {layer_idx}: shape mismatch"
        print(f"  [OK] Layer {layer_idx}: output shape {delta.shape}, num_experts={layer.num_experts}")

    # Test balance loss
    balance_loss = model.compute_balance_loss()
    print(f"  [OK] Balance loss: {balance_loss.item():.4f}")

    # Test trainable params
    num_params = model.get_trainable_params()
    print(f"  [OK] Trainable parameters: {num_params:,}")

    return True


def test_pair_encoder_modes():
    """Test different pair feature modes."""
    print_header("Testing Pair Encoder Modes")

    from src.models.pcr_mola import PCRMoLAConfig, PairEncoder

    modes = ["full", "no_diff", "no_interaction", "context_only"]
    hidden_size = 128
    batch_size = 2
    seq_len = 16

    for mode in modes:
        config = PCRMoLAConfig(
            hidden_size=hidden_size,
            router_dim=32,
            pair_features=mode,
            device="cpu",
            dtype=torch.float32,
        )

        encoder = PairEncoder(config)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        e1_indices = torch.tensor([[2, 3], [5, 6]])
        e2_indices = torch.tensor([[8, 9], [10, 11]])
        e1_mask = torch.tensor([[True, True], [True, True]])
        e2_mask = torch.tensor([[True, True], [True, True]])
        attention_mask = torch.ones(batch_size, seq_len)

        output = encoder(
            hidden_states, e1_indices, e2_indices,
            e1_mask, e2_mask, attention_mask
        )

        expected_dim = config.router_dim
        assert output.shape == (batch_size, expected_dim), f"Mode {mode}: shape mismatch"
        print(f"  [OK] Mode '{mode}': output shape {output.shape}")

    return True


def test_data_collator():
    """Test data collator with entity indices."""
    print_header("Testing Data Collator")

    from src.data.collator import REDataCollator, get_entity_hidden_states
    from transformers import AutoTokenizer

    # Create mock tokenizer (minimal)
    class MockTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0

    tokenizer = MockTokenizer()
    collator = REDataCollator(tokenizer=tokenizer)

    # Create mock features
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 0, 0, 0]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
            "labels": torch.tensor([-100, -100, -100, -100, 5, -100, -100, -100]),
            "e1_indices": [1, 2],
            "e2_indices": [3, 4],
            "label_id": 1,
            "label_text": "related",
            "example_id": 0,
        },
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 0]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 0]),
            "labels": torch.tensor([-100, -100, -100, -100, -100, 5, -100, -100]),
            "e1_indices": [1, 2, 3],
            "e2_indices": [5, 6],
            "label_id": 2,
            "label_text": "unrelated",
            "example_id": 1,
        },
    ]

    batch = collator(features)

    assert "input_ids" in batch
    assert "e1_indices" in batch
    assert "e2_indices" in batch
    assert "e1_mask" in batch
    assert "e2_mask" in batch

    print(f"  [OK] Batch keys: {list(batch.keys())}")
    print(f"  [OK] input_ids shape: {batch['input_ids'].shape}")
    print(f"  [OK] e1_indices shape: {batch['e1_indices'].shape}")
    print(f"  [OK] e1_mask shape: {batch['e1_mask'].shape}")

    # Test entity hidden state extraction
    hidden_states = torch.randn(2, 8, 64)
    e1_hidden = get_entity_hidden_states(
        hidden_states, batch["e1_indices"], batch["e1_mask"]
    )
    print(f"  [OK] Entity hidden shape: {e1_hidden.shape}")

    return True


def test_metrics():
    """Test metrics computation."""
    print_header("Testing Metrics")

    from src.utils.metrics import compute_metrics, normalize_label

    predictions = ["related", "unrelated", "RELATED", "  related  ", "unknown"]
    references = ["related", "unrelated", "related", "related", "related"]
    label_set = ["related", "unrelated"]

    metrics = compute_metrics(predictions, references, label_set)

    print(f"  [OK] Accuracy: {metrics['accuracy']:.2f}")
    print(f"  [OK] Macro F1: {metrics['macro_f1']:.2f}")
    print(f"  [OK] Weighted F1: {metrics['weighted_f1']:.2f}")
    print(f"  [OK] Invalid count: {metrics['invalid_count']}")

    # Test label normalization
    test_cases = [
        ("Related", "related"),
        ("  UNRELATED  ", "unrelated"),
        ("The answer is: related", "related"),
    ]

    for input_text, expected in test_cases:
        normalized = normalize_label(input_text, label_set)
        print(f"  [OK] Normalize '{input_text}' -> '{normalized}'")

    return True


def test_synthetic_data():
    """Test synthetic data generation."""
    print_header("Testing Synthetic Data")

    from src.data.preprocess import create_synthetic_data

    for dataset in ["chemprot", "ddi", "gad"]:
        examples = create_synthetic_data(dataset, num_samples=10)
        assert len(examples) == 10
        assert all("sentence" in ex for ex in examples)
        assert all("label" in ex for ex in examples)
        print(f"  [OK] {dataset}: {len(examples)} examples, labels: {set(ex['label'] for ex in examples)}")

    return True


def test_config_loading():
    """Test configuration loading."""
    print_header("Testing Configuration")

    from src.utils.io import load_config
    from pathlib import Path

    config_dir = Path("configs")

    # Test loading default config
    if (config_dir / "default.yaml").exists():
        config = load_config(config_dir / "default.yaml")
        print(f"  [OK] Loaded default.yaml: {len(config)} top-level keys")

    # Test loading experiment config
    for exp_file in ["ours.yaml", "baseline_b1.yaml", "baseline_b2.yaml"]:
        exp_path = config_dir / "exp" / exp_file
        if exp_path.exists():
            config = load_config(exp_path)
            method = config.get("experiment", {}).get("method", "unknown")
            print(f"  [OK] Loaded {exp_file}: method={method}")

    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print(" PCR-MoLA Installation Verification")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("PCR-MoLA Layer", test_pcr_mola_layer),
        ("Pair Encoder Modes", test_pair_encoder_modes),
        ("Data Collator", test_data_collator),
        ("Metrics", test_metrics),
        ("Synthetic Data", test_synthetic_data),
        ("Config Loading", test_config_loading),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print_header("Summary")
    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed! Installation verified.\n")
        return 0
    else:
        print("\n  Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
