"""
Ensemble Experiment: Neural + Baseline Strategies
==================================================
Tests different ensemble approaches combining neural model with frequency baseline.

Strategies tested:
1. Pure Neural (K=30)
2. Pure Baseline (K=30)
3. Hybrid: Neural for hard parts, Baseline for easy
4. Confidence-weighted: Use neural when confident, baseline otherwise
5. Voting: Include part if either model predicts it (K capped)

Run on RunPod B200 with:
    python ensemble_experiment.py

Outputs:
    outputs/ensemble_experiment/ensemble_results.csv
    outputs/ensemble_experiment/ensemble_report.md

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
HARD_PARTS = [12, 8, 13, 22, 23, 39]
MEDIUM_PARTS = [1, 3, 4, 5, 7, 14, 16, 20, 21, 24, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38]
EASY_PARTS = [2, 6, 9, 10, 11, 15, 17, 18, 19, 25, 26, 28, 29]

# Convert to 0-indexed sets for fast lookup
HARD_PARTS_IDX = set(p - 1 for p in HARD_PARTS)
EASY_PARTS_IDX = set(p - 1 for p in EASY_PARTS)


def load_best_model(checkpoint_path, config):
    """Load model from checkpoint."""
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

    return model


def compute_frequency_baseline(data_path, sequence_length=14):
    """Compute frequency-based baseline predictions."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    parts = df[['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values
    baseline_probs = np.zeros((len(df), 39))

    for i in range(sequence_length, len(df)):
        freq = np.zeros(39)
        for j in range(i - sequence_length, i):
            for p in parts[j]:
                if 1 <= p <= 39:
                    freq[p - 1] += 1
        baseline_probs[i] = freq / sequence_length

    return baseline_probs, df


def compute_tier(hits):
    """Map hits to tier."""
    if hits == 5:
        return 'excellent'
    elif hits == 4:
        return 'good'
    else:
        return 'unacceptable'


class EnsembleStrategy:
    """Base class for ensemble strategies."""

    def __init__(self, pool_size=30):
        self.pool_size = pool_size

    def predict(self, neural_probs, baseline_probs):
        """Return set of predicted part indices (0-indexed)."""
        raise NotImplementedError


class PureNeural(EnsembleStrategy):
    """Use only neural predictions."""

    def predict(self, neural_probs, baseline_probs):
        top_k = np.argsort(neural_probs)[-self.pool_size:]
        return set(top_k)


class PureBaseline(EnsembleStrategy):
    """Use only baseline predictions."""

    def predict(self, neural_probs, baseline_probs):
        top_k = np.argsort(baseline_probs)[-self.pool_size:]
        return set(top_k)


class HybridStrategy(EnsembleStrategy):
    """Neural for hard parts, baseline for easy parts."""

    def predict(self, neural_probs, baseline_probs):
        pool = set()

        # For hard parts: use neural ranking
        hard_neural_scores = [(i, neural_probs[i]) for i in HARD_PARTS_IDX]
        hard_neural_scores.sort(key=lambda x: x[1], reverse=True)

        # For other parts: use baseline ranking
        other_baseline_scores = [(i, baseline_probs[i]) for i in range(39) if i not in HARD_PARTS_IDX]
        other_baseline_scores.sort(key=lambda x: x[1], reverse=True)

        # Fill pool: prioritize neural's top hard parts
        hard_in_pool = 0
        for idx, score in hard_neural_scores:
            if len(pool) < self.pool_size:
                pool.add(idx)
                hard_in_pool += 1

        # Fill rest with baseline's top other parts
        for idx, score in other_baseline_scores:
            if len(pool) >= self.pool_size:
                break
            pool.add(idx)

        return pool


class ConfidenceWeighted(EnsembleStrategy):
    """Use neural when confident (high prob), baseline otherwise."""

    def __init__(self, pool_size=30, confidence_threshold=0.5):
        super().__init__(pool_size)
        self.threshold = confidence_threshold

    def predict(self, neural_probs, baseline_probs):
        # Blend probabilities based on neural confidence
        neural_max = np.max(neural_probs)
        alpha = min(neural_max / self.threshold, 1.0)  # Higher alpha = more neural

        blended = alpha * neural_probs + (1 - alpha) * baseline_probs
        top_k = np.argsort(blended)[-self.pool_size:]
        return set(top_k)


class VotingEnsemble(EnsembleStrategy):
    """Include part if either model ranks it highly, capped at K."""

    def __init__(self, pool_size=30, neural_k=25, baseline_k=25):
        super().__init__(pool_size)
        self.neural_k = neural_k
        self.baseline_k = baseline_k

    def predict(self, neural_probs, baseline_probs):
        neural_top = set(np.argsort(neural_probs)[-self.neural_k:])
        baseline_top = set(np.argsort(baseline_probs)[-self.baseline_k:])

        # Union of both
        candidates = neural_top | baseline_top

        if len(candidates) <= self.pool_size:
            return candidates

        # If too many, rank by combined score
        combined_scores = [(i, neural_probs[i] + baseline_probs[i]) for i in candidates]
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return set(i for i, _ in combined_scores[:self.pool_size])


class AdaptiveHybrid(EnsembleStrategy):
    """Adaptive: more neural weight for hard parts, more baseline for easy."""

    def predict(self, neural_probs, baseline_probs):
        # Weight by category
        blended = np.zeros(39)
        for i in range(39):
            if i in HARD_PARTS_IDX:
                # 70% neural, 30% baseline for hard parts
                blended[i] = 0.7 * neural_probs[i] + 0.3 * baseline_probs[i]
            elif i in EASY_PARTS_IDX:
                # 30% neural, 70% baseline for easy parts
                blended[i] = 0.3 * neural_probs[i] + 0.7 * baseline_probs[i]
            else:
                # 50/50 for medium parts
                blended[i] = 0.5 * neural_probs[i] + 0.5 * baseline_probs[i]

        top_k = np.argsort(blended)[-self.pool_size:]
        return set(top_k)


def run_ensemble_experiment(model, datamodule, baseline_probs, device, pool_size=30):
    """Run all ensemble strategies and compare."""

    strategies = {
        'Pure Neural': PureNeural(pool_size),
        'Pure Baseline': PureBaseline(pool_size),
        'Hybrid (Neural Hard)': HybridStrategy(pool_size),
        'Confidence Weighted': ConfidenceWeighted(pool_size, 0.5),
        'Voting (25+25)': VotingEnsemble(pool_size, 25, 25),
        'Adaptive Hybrid': AdaptiveHybrid(pool_size),
    }

    # Collect results
    results = {name: {'hits': [], 'tier_counts': defaultdict(int)} for name in strategies}

    model.eval()
    model.to(device)
    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            targets = batch['target'].numpy()
            date_indices = batch['date_idx'].numpy()

            # Neural predictions
            logits, _ = model(sequences)
            neural_probs = torch.sigmoid(logits).cpu().numpy()

            batch_size = len(sequences)
            for b in range(batch_size):
                date_idx = date_indices[b]
                actual_parts = set(np.where(targets[b] > 0.5)[0])  # 0-indexed
                base_probs = baseline_probs[date_idx]

                # Test each strategy
                for name, strategy in strategies.items():
                    predicted = strategy.predict(neural_probs[b], base_probs)
                    hits = len(predicted & actual_parts)
                    tier = compute_tier(hits)

                    results[name]['hits'].append(hits)
                    results[name]['tier_counts'][tier] += 1

    # Compute summary statistics
    summary = []
    for name in strategies:
        hits = np.array(results[name]['hits'])
        tier_counts = results[name]['tier_counts']
        total = len(hits)

        excellent_pct = tier_counts['excellent'] / total * 100
        good_pct = tier_counts['good'] / total * 100
        unacceptable_pct = tier_counts['unacceptable'] / total * 100

        summary.append({
            'strategy': name,
            'excellent_pct': excellent_pct,
            'good_pct': good_pct,
            'good_or_better_pct': excellent_pct + good_pct,
            'unacceptable_pct': unacceptable_pct,
            'avg_hits': hits.mean(),
            'total_samples': total
        })

    return pd.DataFrame(summary), results


def generate_ensemble_report(summary_df, output_dir):
    """Generate markdown report."""

    report = []
    report.append("# Ensemble Experiment Results")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nPool Size: K=30")

    report.append("\n## Strategy Comparison\n")
    report.append("| Strategy | Excellent | Good | GoB | Unacceptable | Avg Hits |")
    report.append("|----------|-----------|------|-----|--------------|----------|")

    # Sort by Good-or-Better
    summary_df = summary_df.sort_values('good_or_better_pct', ascending=False)

    best_gob = summary_df['good_or_better_pct'].max()

    for _, row in summary_df.iterrows():
        marker = " **" if row['good_or_better_pct'] == best_gob else ""
        report.append(f"| {row['strategy']}{marker} | {row['excellent_pct']:.1f}% | "
                     f"{row['good_pct']:.1f}% | {row['good_or_better_pct']:.1f}% | "
                     f"{row['unacceptable_pct']:.1f}% | {row['avg_hits']:.2f} |")

    # Analysis
    report.append("\n## Analysis\n")

    baseline_gob = summary_df[summary_df['strategy'] == 'Pure Baseline']['good_or_better_pct'].values[0]
    neural_gob = summary_df[summary_df['strategy'] == 'Pure Neural']['good_or_better_pct'].values[0]
    best_strategy = summary_df.iloc[0]['strategy']
    best_gob_value = summary_df.iloc[0]['good_or_better_pct']

    report.append(f"- **Baseline GoB**: {baseline_gob:.1f}%")
    report.append(f"- **Neural GoB**: {neural_gob:.1f}%")
    report.append(f"- **Neural Lift**: {neural_gob - baseline_gob:+.1f}pp")
    report.append(f"- **Best Strategy**: {best_strategy} ({best_gob_value:.1f}%)")
    report.append(f"- **Best vs Baseline**: {best_gob_value - baseline_gob:+.1f}pp")
    report.append(f"- **Best vs Neural**: {best_gob_value - neural_gob:+.1f}pp")

    # Recommendations
    report.append("\n## Recommendations\n")

    if best_gob_value > neural_gob + 0.5:
        report.append(f"1. **Ensemble outperforms pure neural** by {best_gob_value - neural_gob:.1f}pp")
        report.append(f"2. Recommended strategy: **{best_strategy}**")
        report.append("3. Ensemble captures complementary strengths of both approaches")
    elif neural_gob > baseline_gob + 1.0:
        report.append("1. **Pure neural provides meaningful lift** over baseline")
        report.append("2. Ensemble provides marginal additional benefit")
        report.append("3. Consider pure neural for simplicity if compute allows")
    else:
        report.append("1. **Marginal differences** between strategies")
        report.append("2. Baseline may be sufficient for production")
        report.append("3. Consider operational simplicity over marginal accuracy")

    # Write report
    report_path = output_dir / "ensemble_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    return '\n'.join(report)


def main():
    print("=" * 60)
    print("Ensemble Experiment: Neural + Baseline Strategies")
    print("=" * 60)

    # Setup paths
    base_dir = Path('.')
    data_path = base_dir / 'data' / 'CA5_date.csv'
    config_path = base_dir / 'outputs' / 'best_model' / 'config.yaml'
    checkpoint_dir = base_dir / 'outputs' / 'best_model' / 'checkpoints'
    output_dir = base_dir / 'outputs' / 'ensemble_experiment'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint
    checkpoints = list(checkpoint_dir.glob('best-epoch*.ckpt'))
    if not checkpoints:
        checkpoint_path = checkpoint_dir / 'last.ckpt'
    else:
        checkpoint_path = max(checkpoints, key=lambda p: float(p.stem.split('=')[-1]))

    print(f"Using checkpoint: {checkpoint_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pool_size = config.get('pool_size', 30)
    print(f"Pool size: K={pool_size}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading model...")
    model = load_best_model(checkpoint_path, config)

    # Setup data module
    print("\nSetting up data module...")
    datamodule = CA5DataModule(
        data_path=str(data_path),
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size'],
        num_workers=0,
        val_years=config['val_years'],
        test_years=config['test_years'],
        num_parts=config['num_parts']
    )
    datamodule.setup('test')
    print(f"Test samples: {len(datamodule.test_dataset)}")

    # Compute baseline
    print("\nComputing frequency baseline...")
    baseline_probs, _ = compute_frequency_baseline(str(data_path), config['sequence_length'])

    # Run experiment
    print("\nRunning ensemble experiment...")
    summary_df, detailed_results = run_ensemble_experiment(
        model, datamodule, baseline_probs, device, pool_size
    )

    # Save results
    summary_df.to_csv(output_dir / 'ensemble_results.csv', index=False)

    # Generate report
    print("\n" + "=" * 60)
    report = generate_ensemble_report(summary_df, output_dir)
    print(report)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE:")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("FILES SAVED:")
    print(f"  - {output_dir / 'ensemble_results.csv'}")
    print(f"  - {output_dir / 'ensemble_report.md'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
