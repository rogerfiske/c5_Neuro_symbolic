"""
Production Inference: Hybrid Strategy for CA5 Parts Prediction
===============================================================
Predicts next-day staged parts pool using hybrid strategy:
- Neural model for parts 1-11, 13-39
- Frequency baseline for Part 12 ONLY

Usage:
    python production_inference.py                    # Use most recent 14 days
    python production_inference.py --date 2026-01-20  # Predict for specific date
    python production_inference.py --history 30       # Use 30 days for baseline

Output:
    - Predicted pool of K=30 parts for next day
    - Confidence scores for each part
    - Part 12 handled via baseline (flagged in output)

Author: Dr. Synapse Research Pipeline
Production Version: 1.0 (Hybrid Strategy)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import sys

# Check for PyTorch (may not be available in all environments)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Running in baseline-only mode.")


def load_data(data_path):
    """Load and prepare the CA5 dataset."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_frequency_baseline(df, sequence_length=14, target_date=None):
    """
    Compute frequency-based baseline predictions.

    Args:
        df: DataFrame with historical data
        sequence_length: Number of days to look back
        target_date: Date to predict for (uses last N days before this)

    Returns:
        np.array of shape (39,) with frequencies for each part
    """
    if target_date is not None:
        df = df[df['date'] < target_date]

    # Use last sequence_length days
    recent = df.tail(sequence_length)

    # Count part occurrences
    freq = np.zeros(39)
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        for part_id in recent[col]:
            if 1 <= part_id <= 39:
                freq[part_id - 1] += 1

    # Normalize to get frequencies
    freq = freq / sequence_length
    return freq


def load_neural_model(checkpoint_path, config_path):
    """Load the trained neural model."""
    if not TORCH_AVAILABLE:
        return None

    # Import here to avoid errors if torch not available
    sys.path.insert(0, str(Path(__file__).parent.parent / 'runpod_package'))
    from models.neuro_symbolic import create_model

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            elif not k.startswith('loss_fn'):
                new_state_dict[k] = v

        if new_state_dict:
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, config


def prepare_sequence(df, sequence_length=14, target_date=None):
    """Prepare input sequence for neural model."""
    if target_date is not None:
        df = df[df['date'] < target_date]

    recent = df.tail(sequence_length)

    # Create sequence tensor (sequence_length, 5)
    sequence = recent[['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values
    return sequence


def neural_predict(model, sequence, config):
    """Get neural model predictions."""
    if not TORCH_AVAILABLE or model is None:
        return None

    # Convert to tensor
    sequence_tensor = torch.LongTensor(sequence).unsqueeze(0)  # (1, seq_len, 5)

    with torch.no_grad():
        logits, _ = model(sequence_tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()

    return probs


def hybrid_predict(neural_probs, baseline_probs, pool_size=30):
    """
    Apply hybrid strategy:
    - Use neural for parts 1-11, 13-39
    - Use baseline for Part 12 ONLY

    Args:
        neural_probs: Array of shape (39,) with neural probabilities
        baseline_probs: Array of shape (39,) with baseline frequencies
        pool_size: Number of parts to include in pool (default K=30)

    Returns:
        dict with predicted pool and metadata
    """
    # Start with neural probs, but swap Part 12 (index 11) with baseline
    hybrid_probs = neural_probs.copy() if neural_probs is not None else baseline_probs.copy()

    # Part 12 is at index 11 (0-indexed)
    PART_12_IDX = 11

    if neural_probs is not None:
        hybrid_probs[PART_12_IDX] = baseline_probs[PART_12_IDX]

    # Get top-K parts
    top_k_indices = np.argsort(hybrid_probs)[-pool_size:][::-1]  # Descending order

    # Build result
    pool = []
    for rank, idx in enumerate(top_k_indices):
        part_id = idx + 1  # Convert to 1-indexed
        pool.append({
            'rank': rank + 1,
            'part_id': part_id,
            'probability': hybrid_probs[idx],
            'source': 'baseline' if idx == PART_12_IDX else 'neural',
            'in_pool': True
        })

    # Check if Part 12 made it into the pool
    part_12_in_pool = PART_12_IDX in top_k_indices
    part_12_rank = list(top_k_indices).index(PART_12_IDX) + 1 if part_12_in_pool else None

    # Compute excluded parts (all 39 parts minus those in pool)
    pool_part_ids = set(idx + 1 for idx in top_k_indices)
    all_parts = set(range(1, 40))  # Parts 1-39
    excluded_parts = sorted(all_parts - pool_part_ids)

    return {
        'pool': pool,
        'pool_size': pool_size,
        'part_12_in_pool': part_12_in_pool,
        'part_12_rank': part_12_rank,
        'part_12_baseline_prob': baseline_probs[PART_12_IDX],
        'excluded_parts': excluded_parts,
        'strategy': 'hybrid'
    }


def format_output(result, target_date=None):
    """Format prediction output for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("CA5 PARTS POOL PREDICTION (HYBRID STRATEGY)")
    lines.append("=" * 60)

    if target_date:
        lines.append(f"Prediction Date: {target_date.strftime('%Y-%m-%d')}")
    lines.append(f"Pool Size: K={result['pool_size']}")
    lines.append(f"Strategy: Neural for all parts EXCEPT Part 12 (baseline)")
    lines.append("")

    # Part 12 status
    if result['part_12_in_pool']:
        lines.append(f"[OK] Part 12 IN POOL (rank {result['part_12_rank']}, prob {result['part_12_baseline_prob']:.3f})")
    else:
        lines.append(f"[--] Part 12 NOT in pool (baseline prob {result['part_12_baseline_prob']:.3f})")
    lines.append("")

    # Pool table
    lines.append("PREDICTED POOL:")
    lines.append("-" * 50)
    lines.append(f"{'Rank':>4} | {'Part':>4} | {'Prob':>8} | {'Source':>8}")
    lines.append("-" * 50)

    for item in result['pool']:
        marker = " <--" if item['part_id'] == 12 else ""
        lines.append(f"{item['rank']:>4} | {item['part_id']:>4} | {item['probability']:>8.4f} | {item['source']:>8}{marker}")

    lines.append("-" * 50)
    lines.append("")

    # Part IDs only (for easy copy)
    part_ids = [str(item['part_id']) for item in result['pool']]
    lines.append("POOL PARTS (copy-paste ready):")
    lines.append(", ".join(part_ids))
    lines.append("")

    # Excluded parts
    lines.append("EXCLUDED PARTS:")
    excluded_ids = [str(p) for p in result['excluded_parts']]
    lines.append(", ".join(excluded_ids))
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='CA5 Parts Pool Prediction (Hybrid Strategy)')
    parser.add_argument('--date', type=str, help='Predict for specific date (YYYY-MM-DD)')
    parser.add_argument('--history', type=int, default=14, help='Days of history for baseline (default: 14)')
    parser.add_argument('--pool-size', type=int, default=30, help='Pool size K (default: 30)')
    parser.add_argument('--baseline-only', action='store_true', help='Use baseline only (no neural)')
    parser.add_argument('--output', type=str, help='Save output to file')
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'raw' / 'CA5_date.csv'
    checkpoint_dir = base_dir / 'outputs' / 'best_model' / 'checkpoints'
    config_path = base_dir / 'outputs' / 'best_model' / 'config.yaml'

    # Load data
    print("Loading data...")
    df = load_data(data_path)

    # Determine target date
    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = df['date'].max() + timedelta(days=1)

    print(f"Predicting for: {target_date.strftime('%Y-%m-%d')}")

    # Compute baseline
    print("Computing frequency baseline...")
    baseline_probs = compute_frequency_baseline(df, args.history, target_date)

    # Load neural model (if available and not baseline-only)
    neural_probs = None
    if not args.baseline_only and TORCH_AVAILABLE and config_path.exists():
        print("Loading neural model...")
        try:
            # Find checkpoint
            checkpoints = list(checkpoint_dir.glob('best-epoch*.ckpt'))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: float(p.stem.split('=')[-1]))
            else:
                checkpoint_path = checkpoint_dir / 'last.ckpt'

            if checkpoint_path.exists():
                model, config = load_neural_model(checkpoint_path, config_path)
                sequence = prepare_sequence(df, config['sequence_length'], target_date)
                neural_probs = neural_predict(model, sequence, config)
                print("Neural model loaded successfully.")
            else:
                print("No checkpoint found. Using baseline only.")
        except Exception as e:
            print(f"Error loading neural model: {e}")
            print("Falling back to baseline only.")

    # Make prediction
    print("Generating hybrid prediction...")
    result = hybrid_predict(neural_probs, baseline_probs, args.pool_size)

    # Format and display output
    output = format_output(result, target_date)
    print(output)

    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Output saved to: {args.output}")

    return result


if __name__ == '__main__':
    main()
