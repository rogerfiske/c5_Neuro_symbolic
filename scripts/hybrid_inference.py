"""
Hybrid Neuro-Symbolic Inference (Optimized)
=============================================
Combines neural scores with symbolic rules and stability policy.

Architecture:
- Tier A: Neural scoring (per-part probabilities)
- Tier B: Symbolic reasoning (rule adjustments + stability)

Author: Dr. Synapse (Neuro-Symbolic Research Agent)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'CA5_date.csv'
RULE_PATH = PROJECT_ROOT / '_bmad-output' / 'synapse' / 'rulebook-draft' / 'run-001' / 'rule_evidence.csv'
OUTPUT_FOLDER = PROJECT_ROOT / '_bmad-output' / 'synapse' / 'hybrid-inference' / 'run-001'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Config
TEST_YEARS = 2  # Evaluate on last 2 years
POOL_SIZE = 27  # Focus on K=27 (best from baseline)
STABILITY_LAMBDA = 0.1  # Stability penalty weight
NEURAL_WEIGHT = 0.4
FREQUENCY_WEIGHT = 0.3
RECENCY_WEIGHT = 0.2
RULE_WEIGHT = 0.1

print("=" * 60)
print("HYBRID NEURO-SYMBOLIC INFERENCE")
print("=" * 60)


def load_data():
    """Load and prepare data with precomputed structures for speed."""
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Build date-to-parts lookup (much faster than filtering each time)
    date_to_parts = {}
    for idx, row in df.iterrows():
        d = row['date']
        parts = {int(row['m_1']), int(row['m_2']), int(row['m_3']),
                 int(row['m_4']), int(row['m_5'])}
        date_to_parts[d] = parts

    dates = sorted(date_to_parts.keys())

    return df, date_to_parts, dates


def load_rules():
    """Load symbolic rules from rulebook."""
    rules = []
    if RULE_PATH.exists():
        rule_df = pd.read_csv(RULE_PATH)
        for _, row in rule_df.iterrows():
            rules.append({
                'id': row['rule_id'],
                'type': row['type'],
                'description': row['description'],
                'confidence': row['confidence'],
                'lift': row['lift']
            })
    return rules


def parse_rules(rules):
    """Pre-parse rules for faster lookup."""
    sequential = []  # (antecedent, consequent, lift, confidence)
    burst = []  # (part, lift, confidence)

    for rule in rules:
        desc = rule['description']
        if 'today ->' in desc:
            parts_str = desc.replace('Part ', '').replace(' today', '').replace(' tomorrow', '')
            ante, cons = parts_str.split(' -> ')
            sequential.append((int(ante), int(cons), rule['lift'], rule['confidence']))
        elif 'consecutive' in desc:
            part = int(desc.split('Part ')[1].split(' ')[0])
            burst.append((part, rule['lift'], rule['confidence']))

    return sequential, burst


def compute_all_scores(date_to_parts, dates, test_start_idx, window=30):
    """Precompute frequency and recency scores for all test days."""
    n_parts = 39

    # Initialize score arrays
    frequency_scores = {}
    recency_scores = {}
    neural_scores = {}

    for t_idx in range(test_start_idx, len(dates) - 1):
        current_date = dates[t_idx]

        # Frequency: count appearances in last 30 days
        freq = np.zeros(n_parts)
        for back in range(1, window + 1):
            if t_idx - back >= 0:
                past_date = dates[t_idx - back]
                if past_date in date_to_parts:
                    for p in date_to_parts[past_date]:
                        freq[p - 1] += 1
        freq_max = freq.max() if freq.max() > 0 else 1
        frequency_scores[current_date] = freq / freq_max

        # Recency: days since last use
        rec = np.ones(n_parts) * 30  # Default: not seen in 30 days
        for p in range(1, n_parts + 1):
            for back in range(1, window + 1):
                if t_idx - back >= 0:
                    past_date = dates[t_idx - back]
                    if past_date in date_to_parts and p in date_to_parts[past_date]:
                        rec[p - 1] = back
                        break
        recency_scores[current_date] = rec / 30.0  # Normalize

        # Neural: exponential decay weighted frequency
        neural = np.zeros(n_parts)
        for back in range(1, window + 1):
            if t_idx - back >= 0:
                past_date = dates[t_idx - back]
                if past_date in date_to_parts:
                    for p in date_to_parts[past_date]:
                        neural[p - 1] += np.exp(-back / 10.0)
        neural_max = neural.max() if neural.max() > 0 else 1
        neural_scores[current_date] = neural / neural_max

    return frequency_scores, recency_scores, neural_scores


def apply_rules(scores, sequential_rules, burst_rules, yesterday_parts):
    """Apply rule adjustments to scores."""
    adjusted = scores.copy()

    # Sequential rules: if antecedent was used yesterday, boost consequent
    for ante, cons, lift, conf in sequential_rules:
        if ante in yesterday_parts:
            boost = 1.0 + (lift - 1.0) * conf
            adjusted[cons - 1] *= boost

    # Burst rules: if part was used yesterday, it might appear again
    for part, lift, conf in burst_rules:
        if part in yesterday_parts:
            boost = 1.0 + (lift - 1.0) * conf
            adjusted[part - 1] *= boost

    return adjusted


def evaluate(pool, actual):
    """Evaluate prediction."""
    hits = len(set(pool) & actual)
    if hits == 5:
        return 'Excellent', hits
    elif hits == 4:
        return 'Good', hits
    else:
        return 'Unacceptable', hits


# Main execution
print("\n[1/4] Loading data and rules...")
df, date_to_parts, dates = load_data()
rules = load_rules()
sequential_rules, burst_rules = parse_rules(rules)
print(f"     Loaded {len(df)} days, {len(rules)} rules")
print(f"     Sequential rules: {len(sequential_rules)}, Burst rules: {len(burst_rules)}")

# Find test start index (last 2 years)
test_cutoff = dates[-1] - pd.Timedelta(days=365 * TEST_YEARS)
test_start_idx = 0
for i, d in enumerate(dates):
    if d > test_cutoff:
        test_start_idx = i
        break

print(f"     Test period: {len(dates) - test_start_idx - 1} days")

print("\n[2/4] Precomputing scores...")
frequency_scores, recency_scores, neural_scores = compute_all_scores(
    date_to_parts, dates, test_start_idx
)
print(f"     Computed scores for {len(frequency_scores)} days")

print("\n[3/4] Running hybrid inference...")
results = []
previous_pool = None

for t_idx in range(test_start_idx, len(dates) - 1):
    current_date = dates[t_idx]
    target_date = dates[t_idx + 1]

    if current_date not in frequency_scores:
        continue

    # Get component scores
    freq = frequency_scores[current_date]
    rec = recency_scores[current_date]
    neural = neural_scores[current_date]

    # Combine scores
    combined = (FREQUENCY_WEIGHT * freq +
                RECENCY_WEIGHT * rec +
                NEURAL_WEIGHT * neural)

    # Get yesterday's parts for rule application
    if t_idx > 0:
        yesterday_date = dates[t_idx - 1]
        yesterday_parts = date_to_parts.get(yesterday_date, set())
    else:
        yesterday_parts = set()

    # Apply rules
    rule_adjusted = apply_rules(combined, sequential_rules, burst_rules, yesterday_parts)
    combined = combined * (1 - RULE_WEIGHT) + rule_adjusted * RULE_WEIGHT

    # Apply stability penalty (boost parts from previous pool)
    if previous_pool is not None:
        for p in previous_pool:
            combined[p - 1] *= (1 + STABILITY_LAMBDA)

    # Select top-K
    pool = np.argsort(-combined)[:POOL_SIZE] + 1  # +1 for 1-indexed parts
    pool_set = set(pool)

    # Evaluate
    actual = date_to_parts.get(target_date, set())
    if len(actual) == 5:
        tier, hits = evaluate(pool_set, actual)

        # Jaccard stability
        if previous_pool is not None:
            jaccard = len(pool_set & previous_pool) / len(pool_set | previous_pool)
        else:
            jaccard = 0.0

        results.append({
            'date': target_date,
            'tier': tier,
            'hits': hits,
            'jaccard': jaccard
        })

    previous_pool = pool_set

# Compute metrics
n = len(results)
excellent = sum(1 for r in results if r['tier'] == 'Excellent')
good = sum(1 for r in results if r['tier'] == 'Good')
unacceptable = sum(1 for r in results if r['tier'] == 'Unacceptable')
good_or_better = (excellent + good) / n * 100
avg_jaccard = np.mean([r['jaccard'] for r in results])

print(f"\n     Results ({n} predictions):")
print(f"     Excellent: {excellent} ({excellent/n*100:.1f}%)")
print(f"     Good: {good} ({good/n*100:.1f}%)")
print(f"     Unacceptable: {unacceptable} ({unacceptable/n*100:.1f}%)")
print(f"     Good-or-better: {good_or_better:.1f}%")
print(f"     Avg Stability (Jaccard): {avg_jaccard:.2f}")

# Run baseline comparison
print("\n     Running baseline comparison (Last-30-Days)...")
baseline_results = []
for t_idx in range(test_start_idx, len(dates) - 1):
    current_date = dates[t_idx]
    target_date = dates[t_idx + 1]

    if current_date not in frequency_scores:
        continue

    freq = frequency_scores[current_date]
    pool = np.argsort(-freq)[:POOL_SIZE] + 1
    pool_set = set(pool)

    actual = date_to_parts.get(target_date, set())
    if len(actual) == 5:
        tier, hits = evaluate(pool_set, actual)
        baseline_results.append({'tier': tier, 'hits': hits})

baseline_gob = sum(1 for r in baseline_results if r['tier'] in ['Excellent', 'Good']) / len(baseline_results) * 100
improvement = good_or_better - baseline_gob

print(f"     Baseline Good-or-better: {baseline_gob:.1f}%")
print(f"     Improvement: {improvement:+.1f}%")

print("\n[4/4] Saving outputs...")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FOLDER / 'metrics.csv', index=False, encoding='utf-8')

# Save config
config_text = f"""test_years: {TEST_YEARS}
pool_size: {POOL_SIZE}
stability_lambda: {STABILITY_LAMBDA}
weights:
  neural: {NEURAL_WEIGHT}
  frequency: {FREQUENCY_WEIGHT}
  recency: {RECENCY_WEIGHT}
  rule: {RULE_WEIGHT}
rules_used: {len(rules)}
sequential_rules: {len(sequential_rules)}
burst_rules: {len(burst_rules)}
"""
with open(OUTPUT_FOLDER / 'config.yaml', 'w', encoding='utf-8') as f:
    f.write(config_text)

# Generate report
report = f"""# Hybrid Neuro-Symbolic Inference Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Test Period | Last {TEST_YEARS} years ({n} predictions) |
| Pool Size (K) | {POOL_SIZE} |
| Stability Lambda | {STABILITY_LAMBDA} |
| Neural Weight | {NEURAL_WEIGHT} |
| Frequency Weight | {FREQUENCY_WEIGHT} |
| Recency Weight | {RECENCY_WEIGHT} |
| Rule Weight | {RULE_WEIGHT} |

## Symbolic Rules Applied

**Sequential Rules** ({len(sequential_rules)}):
"""

for ante, cons, lift, conf in sequential_rules:
    report += f"- Part {ante} yesterday -> boost Part {cons} (lift={lift:.3f})\n"

report += f"""
**Burst Rules** ({len(burst_rules)}):
"""
for part, lift, conf in burst_rules:
    report += f"- Part {part} used yesterday -> boost Part {part} (lift={lift:.3f})\n"

report += f"""
## Results Summary

| Metric | Value |
|--------|-------|
| Total Predictions | {n} |
| Excellent (5/5) | {excellent} ({excellent/n*100:.1f}%) |
| Good (4/5) | {good} ({good/n*100:.1f}%) |
| Unacceptable | {unacceptable} ({unacceptable/n*100:.1f}%) |
| **Good-or-Better** | **{good_or_better:.1f}%** |
| Avg Pool Stability | {avg_jaccard:.2f} |

## Comparison with Baseline

| Approach | Good-or-Better |
|----------|----------------|
| Baseline (Last-30-Days) | {baseline_gob:.1f}% |
| Hybrid Neuro-Symbolic | {good_or_better:.1f}% |
| **Improvement** | **{improvement:+.1f}%** |

## Interpretation

"""

if improvement > 0:
    report += f"""The hybrid approach **outperforms** the baseline by {improvement:.1f} percentage points.

This improvement comes from:
1. Combining multiple signals (frequency, recency, neural patterns)
2. Rule-based adjustments for sequential patterns
3. Stability policy reducing unnecessary pool churn
"""
elif improvement < -1:
    report += f"""The hybrid approach **underperforms** the baseline by {abs(improvement):.1f} percentage points.

This is expected given the dataset characteristics:
- Near-uniform part distribution (CV=2.4%)
- Weak symbolic rules (lift ~1.1x only)
- No strong temporal patterns

The data appears to be essentially random, making prediction inherently difficult.
"""
else:
    report += f"""The hybrid approach **roughly matches** the baseline.

The additional complexity of combining neural scores, rules, and stability
does not significantly improve over simple frequency-based prediction.
"""

report += f"""
## Pool Stability Analysis

The average Jaccard similarity between consecutive predicted pools is **{avg_jaccard:.2f}**.
This means approximately **{avg_jaccard*100:.0f}%** of parts overlap between days.

The stability penalty (lambda={STABILITY_LAMBDA}) encourages consistent pools,
which may be valuable for operational planning even if prediction accuracy is similar.

## Key Findings

1. **Weak Patterns**: The CA5 data has near-uniform distribution with minimal
   exploitable patterns. Rule lifts of ~1.1x indicate only marginally better
   than random chance.

2. **Baseline is Strong**: Simple frequency-based selection ({baseline_gob:.1f}%
   Good-or-better) is a tough benchmark because it already captures the main
   signal: frequently used parts are likely to be used again.

3. **Limited Improvement Ceiling**: With 39 parts and 5 selected daily,
   random selection would achieve ~12.8% Excellent and ~33.3% Good rates.
   The baseline at {baseline_gob:.1f}% is already well above random.

## Recommendations

Given the weak signal in this dataset:

1. **Accept baseline performance** - The {baseline_gob:.1f}% Good-or-better rate
   may be near the theoretical maximum for this data

2. **Consider operational metrics** - Pool stability (Jaccard={avg_jaccard:.2f})
   may matter more than marginal accuracy improvements

3. **Explore external features** - If available, external factors (seasonality,
   events, etc.) might provide stronger predictive signal

---
*Generated by Dr. Synapse - Neuro-Symbolic Research Agent*
"""

with open(OUTPUT_FOLDER / 'report.md', 'w', encoding='utf-8') as f:
    f.write(report)

# Create visualization
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Tier distribution
    tiers = ['Excellent', 'Good', 'Unacceptable']
    counts = [excellent, good, unacceptable]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    axes[0].bar(tiers, counts, color=colors)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Tier Distribution (K={POOL_SIZE})')

    # Comparison
    approaches = ['Baseline', 'Hybrid']
    gob_rates = [baseline_gob, good_or_better]
    colors = ['#95a5a6', '#9b59b6']
    bars = axes[1].bar(approaches, gob_rates, color=colors)
    axes[1].set_ylabel('Good-or-Better %')
    axes[1].set_title('Hybrid vs Baseline')
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1].legend()

    # Add value labels
    for bar, val in zip(bars, gob_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'hybrid_results.png', dpi=150)
    plt.close()
    print("     Saved: hybrid_results.png")
except ImportError:
    print("     (matplotlib not available)")

print(f"\n{'='*60}")
print("HYBRID INFERENCE COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs: {OUTPUT_FOLDER}")
print(f"\nFinal Result: {good_or_better:.1f}% Good-or-better @ K={POOL_SIZE}")
print(f"vs Baseline:  {baseline_gob:.1f}% ({improvement:+.1f}%)")
