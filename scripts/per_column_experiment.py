"""
Per-Column Prediction Experiment
================================
Tests the hypothesis that treating each column (m_1 to m_5) as a separate
prediction problem with column-specific part candidates can improve accuracy.

Approach:
1. Build frequency baseline per column (only considering valid parts for that column)
2. Test stable K values (4, 5, 6, 7, 8) per column
3. Combine predictions with deduplication
4. Compare against global approach

Key Finding (2026-01-27):
- Per-column frequency baseline does NOT outperform global baseline
- Best per-column (K=8): 68.3% GoB vs Global: 68.7% GoB (-0.4pp)
- Deduplication causes ~15-20% part overlap between columns
- Insight should be incorporated into NEURAL model instead (see research_backlog)

Author: Dr. Synapse Research Pipeline
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import product

# ============================================================
# Data Loading and Preparation
# ============================================================

def load_data(data_path):
    """Load and prepare the CA5 dataset."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_column_candidates(df, percentile=100):
    """
    Get valid part candidates for each column.

    Args:
        df: DataFrame with historical data
        percentile: Include parts that cover this percentile of occurrences (default 100 = all)

    Returns:
        dict mapping column name to set of valid part IDs
    """
    candidates = {}
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        if percentile >= 100:
            candidates[col] = set(df[col].unique())
        else:
            freq = df[col].value_counts()
            cumsum = freq.cumsum() / len(df)
            threshold = percentile / 100
            top_parts = set(freq[cumsum <= threshold].index.tolist())
            # Add next part to ensure we cross threshold
            remaining = freq[cumsum > threshold]
            if len(remaining) > 0:
                top_parts.add(remaining.index[0])
            candidates[col] = top_parts
    return candidates


def compute_column_frequencies(df, sequence_length=14, target_idx=None):
    """
    Compute frequency-based probabilities PER COLUMN.

    Args:
        df: DataFrame with historical data
        sequence_length: Days to look back
        target_idx: Index to predict for (uses prior sequence_length days)

    Returns:
        dict mapping column name to dict of {part_id: frequency}
    """
    if target_idx is None:
        target_idx = len(df)

    start_idx = max(0, target_idx - sequence_length)
    recent = df.iloc[start_idx:target_idx]

    col_freqs = {}
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        freq = recent[col].value_counts()
        col_freqs[col] = {part: count / len(recent) for part, count in freq.items()}

    return col_freqs


def compute_global_frequencies(df, sequence_length=14, target_idx=None):
    """Compute global part frequencies (original approach)."""
    if target_idx is None:
        target_idx = len(df)

    start_idx = max(0, target_idx - sequence_length)
    recent = df.iloc[start_idx:target_idx]

    freq = np.zeros(39)
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        for part_id in recent[col]:
            freq[part_id - 1] += 1

    return freq / len(recent)


# ============================================================
# Prediction Strategies
# ============================================================

def predict_global(global_freq, pool_size=30):
    """Original global approach: top K parts overall."""
    top_k = np.argsort(global_freq)[-pool_size:]
    return set(idx + 1 for idx in top_k)  # Convert to 1-indexed


def predict_per_column(col_freqs, candidates, k_per_col, fill_to=30):
    """
    Per-column prediction with deduplication.

    Args:
        col_freqs: dict of {column: {part: freq}}
        candidates: dict of {column: set of valid parts}
        k_per_col: dict of {column: K value} or single int for all
        fill_to: Final pool size after deduplication

    Returns:
        set of predicted part IDs
    """
    if isinstance(k_per_col, int):
        k_per_col = {col: k_per_col for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']}

    # Step 1: Get top-K predictions per column
    per_col_predictions = {}
    per_col_scores = {}  # Track scores for deduplication priority

    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        valid_parts = candidates[col]
        freqs = col_freqs.get(col, {})

        # Score each valid part (0 if not seen recently)
        scored = [(part, freqs.get(part, 0)) for part in valid_parts]
        scored.sort(key=lambda x: -x[1])

        k = k_per_col[col]
        top_k = scored[:k]
        per_col_predictions[col] = [p for p, s in top_k]
        for p, s in top_k:
            if p not in per_col_scores or s > per_col_scores[p]:
                per_col_scores[p] = s

    # Step 2: Combine with deduplication (keep highest-scoring occurrence)
    pool = set()
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        for part in per_col_predictions[col]:
            pool.add(part)

    # Step 3: If pool < fill_to, add more parts based on global scores
    if len(pool) < fill_to:
        # Get all remaining candidates with scores
        all_remaining = []
        for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
            for part in candidates[col]:
                if part not in pool:
                    score = col_freqs.get(col, {}).get(part, 0)
                    all_remaining.append((part, score))

        # Sort by score descending, take what we need
        all_remaining.sort(key=lambda x: -x[1])
        seen = set()
        for part, score in all_remaining:
            if part not in pool and part not in seen:
                pool.add(part)
                seen.add(part)
                if len(pool) >= fill_to:
                    break

    # Step 4: If pool > fill_to, trim based on scores
    if len(pool) > fill_to:
        scored_pool = [(p, per_col_scores.get(p, 0)) for p in pool]
        scored_pool.sort(key=lambda x: -x[1])
        pool = set(p for p, s in scored_pool[:fill_to])

    return pool


# ============================================================
# Evaluation
# ============================================================

def compute_tier(hits):
    """Map hits to tier."""
    if hits == 5:
        return 'excellent'
    elif hits == 4:
        return 'good'
    else:
        return 'unacceptable'


def evaluate_predictions(df, predictions, start_idx):
    """
    Evaluate predictions against actual parts.

    Args:
        df: DataFrame with actual data
        predictions: list of sets (one per day)
        start_idx: Starting index in df for evaluation

    Returns:
        dict with evaluation metrics
    """
    results = {'hits': [], 'tiers': defaultdict(int)}

    for i, pred_pool in enumerate(predictions):
        actual_idx = start_idx + i
        if actual_idx >= len(df):
            break

        actual = set(df.iloc[actual_idx][['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values)
        hits = len(pred_pool & actual)
        tier = compute_tier(hits)

        results['hits'].append(hits)
        results['tiers'][tier] += 1

    total = len(results['hits'])
    results['avg_hits'] = np.mean(results['hits'])
    results['excellent_pct'] = results['tiers']['excellent'] / total * 100
    results['good_pct'] = results['tiers']['good'] / total * 100
    results['gob_pct'] = (results['tiers']['excellent'] + results['tiers']['good']) / total * 100
    results['unacceptable_pct'] = results['tiers']['unacceptable'] / total * 100

    return results


# ============================================================
# K Optimization
# ============================================================

def optimize_k_per_column(df, val_start, val_end, candidates, sequence_length=14):
    """
    Find optimal K for each column using validation data.

    Tests various K combinations and finds the best based on GoB rate.
    """
    print("\n## Optimizing K per Column\n")

    # Test K values from 3 to 15 per column
    k_range = range(3, 16)

    # For efficiency, first optimize each column independently
    best_k = {}

    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        print(f"Optimizing {col}...")
        best_score = -1
        best_k_val = 6

        for k in k_range:
            # Test with this K for current column, K=6 for others
            test_k = {c: 6 for c in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']}
            test_k[col] = k

            predictions = []
            for idx in range(val_start, val_end):
                col_freqs = compute_column_frequencies(df, sequence_length, idx)
                pred = predict_per_column(col_freqs, candidates, test_k, fill_to=30)
                predictions.append(pred)

            results = evaluate_predictions(df, predictions, val_start)

            if results['gob_pct'] > best_score:
                best_score = results['gob_pct']
                best_k_val = k

        best_k[col] = best_k_val
        print(f"  {col}: K={best_k_val} (GoB={best_score:.1f}%)")

    # Now do a local search around the best values
    print("\nRefining with local search...")

    best_overall_k = best_k.copy()
    best_overall_score = -1

    # Test combinations near the best values
    deltas = [-1, 0, 1]
    search_space = list(product(deltas, repeat=5))

    for delta_combo in search_space:
        test_k = {
            'm_1': max(1, best_k['m_1'] + delta_combo[0]),
            'm_2': max(1, best_k['m_2'] + delta_combo[1]),
            'm_3': max(1, best_k['m_3'] + delta_combo[2]),
            'm_4': max(1, best_k['m_4'] + delta_combo[3]),
            'm_5': max(1, best_k['m_5'] + delta_combo[4]),
        }

        predictions = []
        for idx in range(val_start, val_end):
            col_freqs = compute_column_frequencies(df, sequence_length, idx)
            pred = predict_per_column(col_freqs, candidates, test_k, fill_to=30)
            predictions.append(pred)

        results = evaluate_predictions(df, predictions, val_start)

        if results['gob_pct'] > best_overall_score:
            best_overall_score = results['gob_pct']
            best_overall_k = test_k.copy()

    print(f"\nBest K configuration:")
    for col, k in best_overall_k.items():
        print(f"  {col}: K={k}")
    print(f"  Validation GoB: {best_overall_score:.1f}%")

    return best_overall_k


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("PER-COLUMN PREDICTION EXPERIMENT")
    print("Testing column-specific part candidates with optimized K values")
    print("=" * 70)

    # Setup
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'raw' / 'CA5_date.csv'
    output_dir = base_dir / 'outputs' / 'per_column_experiment'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data(data_path)
    print(f"Total records: {len(df)}")

    # Define splits (same as main model)
    test_years = 2.0
    val_years = 0.5
    last_date = df['date'].max()
    test_cutoff = last_date - pd.Timedelta(days=int(365 * test_years))
    val_cutoff = test_cutoff - pd.Timedelta(days=int(365 * val_years))

    train_end = df[df['date'] < val_cutoff].index[-1]
    val_start = train_end + 1
    val_end = df[df['date'] < test_cutoff].index[-1]
    test_start = val_end + 1
    test_end = len(df)

    print(f"\nData splits:")
    print(f"  Train: 0 to {train_end} ({train_end + 1} records)")
    print(f"  Val: {val_start} to {val_end} ({val_end - val_start + 1} records)")
    print(f"  Test: {test_start} to {test_end - 1} ({test_end - test_start} records)")

    # Get column candidates
    print("\n## Column Candidate Analysis\n")
    candidates_100 = get_column_candidates(df.iloc[:train_end+1], percentile=100)
    candidates_95 = get_column_candidates(df.iloc[:train_end+1], percentile=95)

    print("Full candidates (100%):")
    for col, parts in candidates_100.items():
        print(f"  {col}: {len(parts)} parts, range [{min(parts)}-{max(parts)}]")

    print("\n95th percentile candidates:")
    for col, parts in candidates_95.items():
        print(f"  {col}: {len(parts)} parts, range [{min(parts)}-{max(parts)}]")

    # Optimize K on validation set using 95% candidates
    sequence_length = 14
    best_k = optimize_k_per_column(df, val_start, val_end, candidates_95, sequence_length)

    # ============================================================
    # Test Set Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    # Strategy 1: Global baseline (original approach)
    print("\n## Strategy 1: Global Baseline (K=30)\n")
    global_predictions = []
    for idx in range(test_start, test_end):
        global_freq = compute_global_frequencies(df, sequence_length, idx)
        pred = predict_global(global_freq, pool_size=30)
        global_predictions.append(pred)

    global_results = evaluate_predictions(df, global_predictions, test_start)
    print(f"Excellent: {global_results['excellent_pct']:.1f}%")
    print(f"Good: {global_results['good_pct']:.1f}%")
    print(f"GoB: {global_results['gob_pct']:.1f}%")
    print(f"Unacceptable: {global_results['unacceptable_pct']:.1f}%")
    print(f"Avg Hits: {global_results['avg_hits']:.2f}")

    # Strategy 2: Per-column with uniform K=6
    print("\n## Strategy 2: Per-Column Uniform K=6\n")
    uniform_predictions = []
    for idx in range(test_start, test_end):
        col_freqs = compute_column_frequencies(df, sequence_length, idx)
        pred = predict_per_column(col_freqs, candidates_95, k_per_col=6, fill_to=30)
        uniform_predictions.append(pred)

    uniform_results = evaluate_predictions(df, uniform_predictions, test_start)
    print(f"Excellent: {uniform_results['excellent_pct']:.1f}%")
    print(f"Good: {uniform_results['good_pct']:.1f}%")
    print(f"GoB: {uniform_results['gob_pct']:.1f}%")
    print(f"Unacceptable: {uniform_results['unacceptable_pct']:.1f}%")
    print(f"Avg Hits: {uniform_results['avg_hits']:.2f}")

    # Strategy 3: Per-column with optimized K
    print("\n## Strategy 3: Per-Column Optimized K\n")
    print(f"Using K values: {best_k}")
    optimized_predictions = []
    for idx in range(test_start, test_end):
        col_freqs = compute_column_frequencies(df, sequence_length, idx)
        pred = predict_per_column(col_freqs, candidates_95, k_per_col=best_k, fill_to=30)
        optimized_predictions.append(pred)

    optimized_results = evaluate_predictions(df, optimized_predictions, test_start)
    print(f"Excellent: {optimized_results['excellent_pct']:.1f}%")
    print(f"Good: {optimized_results['good_pct']:.1f}%")
    print(f"GoB: {optimized_results['gob_pct']:.1f}%")
    print(f"Unacceptable: {optimized_results['unacceptable_pct']:.1f}%")
    print(f"Avg Hits: {optimized_results['avg_hits']:.2f}")

    # Strategy 4: Per-column with 100% candidates (full range)
    print("\n## Strategy 4: Per-Column Full Range (100% candidates)\n")
    full_predictions = []
    for idx in range(test_start, test_end):
        col_freqs = compute_column_frequencies(df, sequence_length, idx)
        pred = predict_per_column(col_freqs, candidates_100, k_per_col=best_k, fill_to=30)
        full_predictions.append(pred)

    full_results = evaluate_predictions(df, full_predictions, test_start)
    print(f"Excellent: {full_results['excellent_pct']:.1f}%")
    print(f"Good: {full_results['good_pct']:.1f}%")
    print(f"GoB: {full_results['gob_pct']:.1f}%")
    print(f"Unacceptable: {full_results['unacceptable_pct']:.1f}%")
    print(f"Avg Hits: {full_results['avg_hits']:.2f}")

    # ============================================================
    # Summary Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n| Strategy | Excellent | Good | GoB | Unacceptable | Avg Hits |")
    print("|----------|-----------|------|-----|--------------|----------|")
    print(f"| Global Baseline | {global_results['excellent_pct']:.1f}% | {global_results['good_pct']:.1f}% | {global_results['gob_pct']:.1f}% | {global_results['unacceptable_pct']:.1f}% | {global_results['avg_hits']:.2f} |")
    print(f"| Per-Col Uniform K=6 | {uniform_results['excellent_pct']:.1f}% | {uniform_results['good_pct']:.1f}% | {uniform_results['gob_pct']:.1f}% | {uniform_results['unacceptable_pct']:.1f}% | {uniform_results['avg_hits']:.2f} |")
    print(f"| Per-Col Optimized K | {optimized_results['excellent_pct']:.1f}% | {optimized_results['good_pct']:.1f}% | {optimized_results['gob_pct']:.1f}% | {optimized_results['unacceptable_pct']:.1f}% | {optimized_results['avg_hits']:.2f} |")
    print(f"| Per-Col Full Range | {full_results['excellent_pct']:.1f}% | {full_results['good_pct']:.1f}% | {full_results['gob_pct']:.1f}% | {full_results['unacceptable_pct']:.1f}% | {full_results['avg_hits']:.2f} |")

    # Find best strategy
    strategies = [
        ('Global Baseline', global_results),
        ('Per-Col Uniform K=6', uniform_results),
        ('Per-Col Optimized K', optimized_results),
        ('Per-Col Full Range', full_results),
    ]
    best_strategy = max(strategies, key=lambda x: x[1]['gob_pct'])

    print(f"\n**Best Strategy**: {best_strategy[0]} with {best_strategy[1]['gob_pct']:.1f}% GoB")

    baseline_gob = global_results['gob_pct']
    best_gob = best_strategy[1]['gob_pct']
    print(f"**Improvement over Global Baseline**: {best_gob - baseline_gob:+.1f}pp")

    # ============================================================
    # Part 12 Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 12 ANALYSIS (Per-Column Approach)")
    print("=" * 70)

    # Check Part 12 recall in optimized strategy
    part_12_needed = 0
    part_12_captured_global = 0
    part_12_captured_percol = 0

    for i, idx in enumerate(range(test_start, test_end)):
        actual = set(df.iloc[idx][['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values)
        if 12 in actual:
            part_12_needed += 1
            if 12 in global_predictions[i]:
                part_12_captured_global += 1
            if 12 in optimized_predictions[i]:
                part_12_captured_percol += 1

    print(f"\nPart 12 occurrences in test set: {part_12_needed}")
    print(f"Part 12 recall (Global Baseline): {part_12_captured_global}/{part_12_needed} ({part_12_captured_global/part_12_needed*100:.1f}%)")
    print(f"Part 12 recall (Per-Column Opt): {part_12_captured_percol}/{part_12_needed} ({part_12_captured_percol/part_12_needed*100:.1f}%)")

    # Which columns contain Part 12?
    print("\nPart 12 column distribution (training data):")
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        count = (df.iloc[:train_end+1][col] == 12).sum()
        if count > 0:
            print(f"  {col}: {count} occurrences ({count/(train_end+1)*100:.2f}%)")

    # ============================================================
    # Save Results
    # ============================================================
    results_df = pd.DataFrame([
        {'strategy': 'Global Baseline', **{k: v for k, v in global_results.items() if k != 'hits' and k != 'tiers'}},
        {'strategy': 'Per-Col Uniform K=6', **{k: v for k, v in uniform_results.items() if k != 'hits' and k != 'tiers'}},
        {'strategy': 'Per-Col Optimized K', **{k: v for k, v in optimized_results.items() if k != 'hits' and k != 'tiers'}},
        {'strategy': 'Per-Col Full Range', **{k: v for k, v in full_results.items() if k != 'hits' and k != 'tiers'}},
    ])
    results_df.to_csv(output_dir / 'per_column_results.csv', index=False)

    # Save optimal K configuration
    with open(output_dir / 'optimal_k_config.txt', 'w') as f:
        f.write("Optimal K per Column\n")
        f.write("=" * 30 + "\n")
        for col, k in best_k.items():
            f.write(f"{col}: K={k}\n")
        f.write(f"\nTotal K sum: {sum(best_k.values())}\n")
        f.write(f"Expected pool before dedup: {sum(best_k.values())}\n")
        f.write(f"Final pool size: 30\n")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
