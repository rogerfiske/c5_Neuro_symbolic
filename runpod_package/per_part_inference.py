"""
Per-Part Inference Analysis
===========================
Loads best model, runs inference on test set, and compares neural vs baseline
specifically on hard parts to determine where neural value comes from.

Run on RunPod B200 with:
    python per_part_inference.py

Outputs:
    outputs/per_part_analysis/per_part_results.csv
    outputs/per_part_analysis/neural_vs_baseline_by_category.csv
    outputs/per_part_analysis/analysis_report.md

Author: Dr. Synapse Research Pipeline
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
from collections import defaultdict

from models.neuro_symbolic import create_model
from data_module import CA5DataModule


# Part categories from Phase 2 analysis
HARD_PARTS = [12, 8, 13, 22, 23, 39]  # <70% recall at K=30
MEDIUM_PARTS = [1, 3, 4, 5, 7, 14, 16, 20, 21, 24, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38]
EASY_PARTS = [2, 6, 9, 10, 11, 15, 17, 18, 19, 25, 26, 28, 29]  # >80% recall


def load_best_model(checkpoint_path, config):
    """Load model from checkpoint."""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle Lightning checkpoint format
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning wrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            elif not k.startswith('loss_fn'):
                new_state_dict[k] = v

        if new_state_dict:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {len(new_state_dict)} parameters from checkpoint")
        else:
            # Fall back to direct loading
            model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    return model


def compute_frequency_baseline(data_path, sequence_length=14):
    """Compute frequency-based baseline predictions."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    parts = df[['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values

    # Compute rolling frequency for each day
    baseline_probs = np.zeros((len(df), 39))

    for i in range(sequence_length, len(df)):
        # Count parts in previous 'sequence_length' days
        freq = np.zeros(39)
        for j in range(i - sequence_length, i):
            for p in parts[j]:
                if 1 <= p <= 39:
                    freq[p - 1] += 1
        # Normalize to probabilities
        baseline_probs[i] = freq / sequence_length

    return baseline_probs, df


def run_inference(model, datamodule, device, pool_size=30):
    """Run inference on test set and collect per-prediction results."""
    model.eval()
    model.to(device)

    test_loader = datamodule.test_dataloader()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            targets = batch['target'].numpy()
            date_indices = batch['date_idx'].numpy()

            # Neural predictions
            logits, aux = model(sequences)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Get top-K predictions
            top_k_indices = np.argsort(probs, axis=1)[:, -pool_size:]

            batch_size = len(sequences)
            for b in range(batch_size):
                # Which parts were actually needed
                actual_parts = np.where(targets[b] > 0.5)[0] + 1  # 1-indexed

                # Which parts were predicted (in pool)
                predicted_parts = top_k_indices[b] + 1  # 1-indexed

                # Per-part analysis
                for part_id in actual_parts:
                    part_idx = part_id - 1
                    results.append({
                        'date_idx': date_indices[b],
                        'part_id': part_id,
                        'neural_prob': probs[b, part_idx],
                        'neural_rank': 39 - np.searchsorted(np.sort(probs[b]), probs[b, part_idx]),
                        'in_neural_pool': part_id in predicted_parts,
                        'actual': True
                    })

    return pd.DataFrame(results)


def analyze_results(results_df, baseline_probs, datamodule, pool_size=30):
    """Compare neural vs baseline by part category."""

    # Add baseline predictions to results
    results_df['baseline_prob'] = results_df.apply(
        lambda row: baseline_probs[int(row['date_idx']), int(row['part_id']) - 1],
        axis=1
    )

    # Compute baseline ranks
    def get_baseline_rank(row):
        probs = baseline_probs[int(row['date_idx'])]
        part_prob = probs[int(row['part_id']) - 1]
        return 39 - np.searchsorted(np.sort(probs), part_prob)

    results_df['baseline_rank'] = results_df.apply(get_baseline_rank, axis=1)
    results_df['in_baseline_pool'] = results_df['baseline_rank'] <= pool_size

    # Categorize parts
    def categorize(part_id):
        if part_id in HARD_PARTS:
            return 'hard'
        elif part_id in EASY_PARTS:
            return 'easy'
        else:
            return 'medium'

    results_df['category'] = results_df['part_id'].apply(categorize)

    # Aggregate by category
    category_stats = []
    for category in ['hard', 'medium', 'easy']:
        cat_df = results_df[results_df['category'] == category]
        n_parts = len(set(cat_df['part_id']))
        n_occurrences = len(cat_df)

        neural_recall = cat_df['in_neural_pool'].mean() * 100
        baseline_recall = cat_df['in_baseline_pool'].mean() * 100

        # Cases where neural caught it but baseline missed
        neural_wins = ((cat_df['in_neural_pool']) & (~cat_df['in_baseline_pool'])).sum()
        # Cases where baseline caught it but neural missed
        baseline_wins = ((~cat_df['in_neural_pool']) & (cat_df['in_baseline_pool'])).sum()

        category_stats.append({
            'category': category,
            'n_parts': n_parts,
            'n_occurrences': n_occurrences,
            'neural_recall': neural_recall,
            'baseline_recall': baseline_recall,
            'neural_lift': neural_recall - baseline_recall,
            'neural_wins': neural_wins,
            'baseline_wins': baseline_wins,
            'net_neural_advantage': neural_wins - baseline_wins
        })

    return pd.DataFrame(category_stats), results_df


def generate_report(category_stats, results_df, output_dir):
    """Generate markdown analysis report."""

    report = []
    report.append("# Per-Part Neural vs Baseline Analysis")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nPool Size: K=30")

    report.append("\n## Summary by Part Category\n")
    report.append("| Category | Parts | Occurrences | Neural Recall | Baseline Recall | Neural Lift | Neural Wins | Baseline Wins | Net Advantage |")
    report.append("|----------|-------|-------------|---------------|-----------------|-------------|-------------|---------------|---------------|")

    for _, row in category_stats.iterrows():
        report.append(f"| {row['category'].upper()} | {row['n_parts']} | {row['n_occurrences']} | "
                     f"{row['neural_recall']:.1f}% | {row['baseline_recall']:.1f}% | "
                     f"{row['neural_lift']:+.1f}pp | {row['neural_wins']} | {row['baseline_wins']} | "
                     f"{row['net_neural_advantage']:+d} |")

    # Total row
    total_neural = results_df['in_neural_pool'].mean() * 100
    total_baseline = results_df['in_baseline_pool'].mean() * 100
    total_neural_wins = ((results_df['in_neural_pool']) & (~results_df['in_baseline_pool'])).sum()
    total_baseline_wins = ((~results_df['in_neural_pool']) & (results_df['in_baseline_pool'])).sum()

    report.append(f"| **TOTAL** | 39 | {len(results_df)} | "
                 f"{total_neural:.1f}% | {total_baseline:.1f}% | "
                 f"{total_neural-total_baseline:+.1f}pp | {total_neural_wins} | {total_baseline_wins} | "
                 f"{total_neural_wins-total_baseline_wins:+d} |")

    report.append("\n## Key Findings\n")

    # Analyze where neural excels
    hard_stats = category_stats[category_stats['category'] == 'hard'].iloc[0]
    easy_stats = category_stats[category_stats['category'] == 'easy'].iloc[0]

    if hard_stats['neural_lift'] > easy_stats['neural_lift']:
        report.append("**CONFIRMED: Neural model provides greater lift on HARD parts.**\n")
        report.append(f"- Hard parts neural lift: {hard_stats['neural_lift']:+.1f}pp")
        report.append(f"- Easy parts neural lift: {easy_stats['neural_lift']:+.1f}pp")
        report.append(f"- Differential: {hard_stats['neural_lift'] - easy_stats['neural_lift']:.1f}pp more lift on hard parts")
    else:
        report.append("**Neural model provides similar or less lift on hard parts vs easy parts.**\n")
        report.append(f"- Hard parts neural lift: {hard_stats['neural_lift']:+.1f}pp")
        report.append(f"- Easy parts neural lift: {easy_stats['neural_lift']:+.1f}pp")

    report.append("\n## Per-Part Breakdown (Hard Parts)\n")
    report.append("| Part ID | Occurrences | Neural Recall | Baseline Recall | Lift |")
    report.append("|---------|-------------|---------------|-----------------|------|")

    for part_id in HARD_PARTS:
        part_df = results_df[results_df['part_id'] == part_id]
        if len(part_df) > 0:
            neural_recall = part_df['in_neural_pool'].mean() * 100
            baseline_recall = part_df['in_baseline_pool'].mean() * 100
            report.append(f"| {part_id} | {len(part_df)} | {neural_recall:.1f}% | "
                         f"{baseline_recall:.1f}% | {neural_recall-baseline_recall:+.1f}pp |")

    report.append("\n## Recommendations\n")

    if hard_stats['neural_lift'] > 2.0:
        report.append("1. **Ensemble Strategy Viable**: Neural model adds meaningful value on hard parts")
        report.append("2. **Proposed Ensemble**: Use neural predictions for hard parts, baseline for easy parts")
        report.append("3. **Expected Benefit**: Reduced compute while maintaining neural advantage where it matters")
    else:
        report.append("1. Neural lift is marginal across all categories")
        report.append("2. Baseline frequency method may be sufficient for production")
        report.append("3. Consider simplicity vs marginal accuracy tradeoff")

    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {report_path}")
    return '\n'.join(report)


def main():
    print("=" * 60)
    print("Per-Part Neural vs Baseline Analysis")
    print("=" * 60)

    # Setup paths
    base_dir = Path('.')
    data_path = base_dir / 'data' / 'CA5_date.csv'
    config_path = base_dir / 'outputs' / 'best_model' / 'config.yaml'
    checkpoint_dir = base_dir / 'outputs' / 'best_model' / 'checkpoints'
    output_dir = base_dir / 'outputs' / 'per_part_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint
    checkpoints = list(checkpoint_dir.glob('best-epoch*.ckpt'))
    if not checkpoints:
        # Fall back to last.ckpt
        checkpoint_path = checkpoint_dir / 'last.ckpt'
    else:
        # Get the one with highest val metric
        checkpoint_path = max(checkpoints, key=lambda p: float(p.stem.split('=')[-1]))

    print(f"Using checkpoint: {checkpoint_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config: embed_dim={config['embed_dim']}, hidden_dim={config['hidden_dim']}, "
          f"num_layers={config['num_layers']}, pool_size={config['pool_size']}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading model...")
    model = load_best_model(checkpoint_path, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data module
    print("\nSetting up data module...")
    datamodule = CA5DataModule(
        data_path=str(data_path),
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size'],
        num_workers=0,  # For simplicity on RunPod
        val_years=config['val_years'],
        test_years=config['test_years'],
        num_parts=config['num_parts']
    )
    datamodule.setup('test')
    print(f"Test samples: {len(datamodule.test_dataset)}")

    # Compute baseline
    print("\nComputing frequency baseline...")
    baseline_probs, df = compute_frequency_baseline(str(data_path), config['sequence_length'])

    # Run neural inference
    print("\nRunning neural inference...")
    results_df = run_inference(model, datamodule, device, config['pool_size'])
    print(f"Collected {len(results_df)} part occurrences")

    # Analyze results
    print("\nAnalyzing results...")
    category_stats, full_results = analyze_results(
        results_df, baseline_probs, datamodule, config['pool_size']
    )

    # Save results
    full_results.to_csv(output_dir / 'per_part_results.csv', index=False)
    category_stats.to_csv(output_dir / 'neural_vs_baseline_by_category.csv', index=False)

    # Generate report
    print("\n" + "=" * 60)
    report = generate_report(category_stats, full_results, output_dir)
    print(report)

    # Print summary
    print("\n" + "=" * 60)
    print("FILES SAVED:")
    print(f"  - {output_dir / 'per_part_results.csv'}")
    print(f"  - {output_dir / 'neural_vs_baseline_by_category.csv'}")
    print(f"  - {output_dir / 'analysis_report.md'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
