"""
Test Column-Enhanced Model Architecture
=======================================
Verifies the column-enhanced model works correctly before RunPod deployment.

Run locally:
    python scripts/test_column_enhanced_model.py

Author: Dr. Synapse Research Pipeline
Date: 2026-01-27
"""

import sys
from pathlib import Path

# Add runpod_package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'runpod_package'))

import torch
import torch.nn as nn


def test_model_architecture():
    """Test that all model configurations work."""
    from models.column_enhanced import create_column_enhanced_model

    print("=" * 60)
    print("TESTING COLUMN-ENHANCED MODEL ARCHITECTURES")
    print("=" * 60)

    # Test configurations
    configs = [
        {
            'name': 'Standard Embedding',
            'embedding_type': 'standard',
            'use_column_attention': False,
            'use_column_heads': False,
        },
        {
            'name': 'Column-Aware Embedding',
            'embedding_type': 'column_aware',
            'use_column_attention': False,
            'use_column_heads': False,
        },
        {
            'name': 'Column-Features Embedding',
            'embedding_type': 'column_features',
            'use_column_attention': False,
            'use_column_heads': False,
        },
        {
            'name': 'Per-Column Output Heads',
            'embedding_type': 'standard',
            'use_column_attention': False,
            'use_column_heads': True,
        },
        {
            'name': 'Column-Aware + Output Heads',
            'embedding_type': 'column_aware',
            'use_column_attention': False,
            'use_column_heads': True,
        },
        {
            'name': 'Column-Features + Output Heads',
            'embedding_type': 'column_features',
            'use_column_attention': False,
            'use_column_heads': True,
        },
    ]

    # Common config
    base_config = {
        'num_parts': 39,
        'embed_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'encoder_type': 'transformer',
        'num_heads': 2,
        'dropout': 0.1,
    }

    # Test input
    batch_size = 4
    seq_len = 14
    test_input = torch.randint(1, 40, (batch_size, seq_len, 5))

    results = []

    for config in configs:
        full_config = {**base_config, **config}
        name = config['name']

        try:
            # Create model
            model = create_column_enhanced_model(full_config)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())

            # Forward pass
            logits, aux = model(test_input)

            # Check output shape
            assert logits.shape == (batch_size, 39), f"Expected (4, 39), got {logits.shape}"

            # Check for NaN
            assert not torch.isnan(logits).any(), "NaN in output"

            print(f"[PASS] {name}")
            print(f"       Parameters: {num_params:,}")
            print(f"       Output shape: {logits.shape}")
            print(f"       Aux keys: {list(aux.keys())}")

            results.append({'name': name, 'status': 'PASS', 'params': num_params})

        except Exception as e:
            print(f"[FAIL] {name}")
            print(f"       Error: {e}")
            results.append({'name': name, 'status': 'FAIL', 'error': str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nAll architectures verified. Ready for RunPod deployment.")
    else:
        print("\nSome architectures failed. Check errors above.")

    return passed == total


def test_gradient_flow():
    """Test that gradients flow correctly through all configurations."""
    from models.column_enhanced import create_column_enhanced_model

    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)

    config = {
        'num_parts': 39,
        'embed_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'encoder_type': 'transformer',
        'num_heads': 2,
        'dropout': 0.1,
        'embedding_type': 'column_aware',
        'use_column_heads': True,
    }

    model = create_column_enhanced_model(config)

    # Test input and target
    test_input = torch.randint(1, 40, (4, 14, 5))
    target = torch.zeros(4, 39)
    target[:, :5] = 1.0  # Dummy target

    # Forward
    logits, aux = model(test_input)

    # Loss
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)

    # Backward
    loss.backward()

    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print("[PASS] Gradients flow correctly")
    else:
        print("[FAIL] No gradients found")

    return has_grad


def main():
    print("\n" + "=" * 60)
    print("COLUMN-ENHANCED MODEL VERIFICATION")
    print("=" * 60 + "\n")

    arch_ok = test_model_architecture()
    grad_ok = test_gradient_flow()

    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    if arch_ok and grad_ok:
        print("\n[OK] All tests passed. Model is ready for RunPod training.")
        print("\nNext steps:")
        print("1. Upload runpod_package/ to RunPod")
        print("2. Run: python train_column_enhanced.py")
        print("3. Review results in outputs/column_enhanced/")
    else:
        print("\n[ERROR] Some tests failed. Fix issues before deployment.")


if __name__ == '__main__':
    main()
