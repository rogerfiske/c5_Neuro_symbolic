"""
Baseline Suite for CA5 Dataset
==============================
Run ID: run-001
Purpose: Establish performance floor with simple methods before advanced models

What this script does (in plain English):
1. Loads the CA5 dataset
2. Tests 5 simple prediction methods (baselines)
3. For each method, tries pool sizes K = 20 to 27
4. Measures: How often do we catch all 5 parts? 4 of 5? etc.
5. Finds the best simple method as our benchmark to beat

To run: python scripts/baseline_suite.py
Estimated time: 2-5 minutes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RUN_ID = "run-001"
SEED = 42
np.random.seed(SEED)

# Pool sizes to test
K_VALUES = list(range(20, 28))  # 20, 21, 22, ..., 27

# Rolling window sizes for Last-N-Days baseline
WINDOW_SIZES = [7, 14, 30]

# Train/test split: use last 2 years for testing
TEST_YEARS = 2

# Paths
PROJECT_ROOT = Path("C:/Users/Minis/CascadeProjects/c5_neuro_symbolic")
DATA_PATH = PROJECT_ROOT / "data/raw/CA5_date.csv"
OUTPUT_FOLDER = PROJECT_ROOT / f"_bmad-output/synapse/baseline-suite/{RUN_ID}"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BASELINE SUITE - CA5 PREDICTIVE MAINTENANCE")
print(f"Run ID: {RUN_ID}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading dataset...")

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
part_cols = ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']

print(f"  Total records: {len(df):,}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Split into train/test
cutoff_date = df['date'].max() - pd.Timedelta(days=365 * TEST_YEARS)
train_df = df[df['date'] < cutoff_date].copy()
test_df = df[df['date'] >= cutoff_date].copy()

print(f"  Training: {len(train_df):,} records (up to {cutoff_date.date()})")
print(f"  Testing: {len(test_df):,} records ({TEST_YEARS} years)")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_actual_parts(row):
    """Get the 5 actual parts for a day as a set."""
    return set(row[part_cols].values)

def compute_tier(actual_parts, predicted_pool):
    """
    Compute tier for a single prediction.
    actual_parts: set of 5 actual parts
    predicted_pool: set of K predicted parts
    Returns: 'excellent', 'good', or 'unacceptable'
    """
    hits = len(actual_parts & predicted_pool)
    if hits == 5:
        return 'excellent'
    elif hits == 4:
        return 'good'
    else:
        return 'unacceptable'

def evaluate_baseline(predictions, test_data, k_values):
    """
    Evaluate a baseline across multiple K values.
    predictions: dict mapping date -> ranked list of parts (most likely first)
    test_data: DataFrame with actual parts
    Returns: dict of metrics per K
    """
    results = {}

    for k in k_values:
        tiers = []
        for idx, row in test_data.iterrows():
            date = row['date']
            actual = get_actual_parts(row)

            if date in predictions:
                ranked_parts = predictions[date]
                pool = set(ranked_parts[:k])  # Top K parts
                tier = compute_tier(actual, pool)
                tiers.append(tier)

        if tiers:
            n = len(tiers)
            excellent = tiers.count('excellent') / n
            good = tiers.count('good') / n
            unacceptable = tiers.count('unacceptable') / n
            good_or_better = excellent + good

            results[k] = {
                'excellent_rate': excellent,
                'good_rate': good,
                'good_or_better_rate': good_or_better,
                'unacceptable_rate': unacceptable,
                'n_predictions': n
            }

    return results

# ============================================================
# BASELINE 1: GLOBAL FREQUENCY
# ============================================================
print("\n[2/6] Building baselines...")
print("  [2a] Frequency baseline (most common parts overall)...")

# Count global frequency from training data
all_train_parts = train_df[part_cols].values.flatten()
freq_counts = pd.Series(all_train_parts).value_counts()
freq_ranked = freq_counts.index.tolist()  # Sorted by frequency (descending)

# Predict same ranking for all test days
freq_predictions = {row['date']: freq_ranked for _, row in test_df.iterrows()}
freq_results = evaluate_baseline(freq_predictions, test_df, K_VALUES)

best_k_freq = max(freq_results.keys(), key=lambda k: freq_results[k]['good_or_better_rate'])
print(f"    Best K={best_k_freq}: Good-or-better = {freq_results[best_k_freq]['good_or_better_rate']:.1%}")

# ============================================================
# BASELINE 2: RECENCY (Parts due for use)
# ============================================================
print("  [2b] Recency baseline (parts not used recently)...")

recency_predictions = {}
last_used = {p: pd.Timestamp('1990-01-01') for p in range(1, 40)}  # Initialize

for idx, row in df.iterrows():
    date = row['date']
    actual = get_actual_parts(row)

    # For test dates, make prediction based on last_used
    if date >= cutoff_date:
        # Rank by days since last use (more days = higher priority)
        days_since = {p: (date - last_used[p]).days for p in range(1, 40)}
        ranked = sorted(days_since.keys(), key=lambda p: -days_since[p])
        recency_predictions[date] = ranked

    # Update last_used
    for p in actual:
        last_used[p] = date

recency_results = evaluate_baseline(recency_predictions, test_df, K_VALUES)
best_k_recency = max(recency_results.keys(), key=lambda k: recency_results[k]['good_or_better_rate'])
print(f"    Best K={best_k_recency}: Good-or-better = {recency_results[best_k_recency]['good_or_better_rate']:.1%}")

# ============================================================
# BASELINE 3: LAST-N-DAYS FREQUENCY
# ============================================================
print("  [2c] Last-N-Days baseline (recent frequency)...")

window_results = {}
for window in WINDOW_SIZES:
    window_predictions = {}

    for idx, row in test_df.iterrows():
        date = row['date']

        # Get data from last N days
        window_start = date - pd.Timedelta(days=window)
        window_data = df[(df['date'] >= window_start) & (df['date'] < date)]

        if len(window_data) > 0:
            window_parts = window_data[part_cols].values.flatten()
            window_freq = pd.Series(window_parts).value_counts()
            # Add parts with 0 frequency
            for p in range(1, 40):
                if p not in window_freq.index:
                    window_freq[p] = 0
            ranked = window_freq.sort_values(ascending=False).index.tolist()
        else:
            ranked = freq_ranked  # Fallback to global frequency

        window_predictions[date] = ranked

    window_results[window] = evaluate_baseline(window_predictions, test_df, K_VALUES)

# Find best window
best_window = None
best_window_score = 0
for window, results in window_results.items():
    best_k = max(results.keys(), key=lambda k: results[k]['good_or_better_rate'])
    score = results[best_k]['good_or_better_rate']
    if score > best_window_score:
        best_window_score = score
        best_window = window

print(f"    Best window={best_window} days: Good-or-better = {best_window_score:.1%}")

# Use best window for final results
lastn_results = window_results[best_window]
lastn_predictions = {}  # Rebuild for stability analysis later

# ============================================================
# BASELINE 4: CO-OCCURRENCE
# ============================================================
print("  [2d] Co-occurrence baseline (part associations)...")

# Build co-occurrence matrix from training data
cooccur = defaultdict(lambda: defaultdict(int))
for _, row in train_df.iterrows():
    parts = list(get_actual_parts(row))
    for i, p1 in enumerate(parts):
        for p2 in parts[i+1:]:
            cooccur[p1][p2] += 1
            cooccur[p2][p1] += 1

# For each test day, rank by association with recent parts
cooccur_predictions = {}
recent_window = 7  # Look at last 7 days

for idx, row in test_df.iterrows():
    date = row['date']

    # Get parts from recent days
    recent_start = date - pd.Timedelta(days=recent_window)
    recent_data = df[(df['date'] >= recent_start) & (df['date'] < date)]

    if len(recent_data) > 0:
        recent_parts = set(recent_data[part_cols].values.flatten())

        # Score each part by co-occurrence with recent parts
        scores = {}
        for p in range(1, 40):
            score = sum(cooccur[p][rp] for rp in recent_parts)
            scores[p] = score

        ranked = sorted(scores.keys(), key=lambda p: -scores[p])
    else:
        ranked = freq_ranked

    cooccur_predictions[date] = ranked

cooccur_results = evaluate_baseline(cooccur_predictions, test_df, K_VALUES)
best_k_cooccur = max(cooccur_results.keys(), key=lambda k: cooccur_results[k]['good_or_better_rate'])
print(f"    Best K={best_k_cooccur}: Good-or-better = {cooccur_results[best_k_cooccur]['good_or_better_rate']:.1%}")

# ============================================================
# BASELINE 5: COMBINED (Frequency + Recency)
# ============================================================
print("  [2e] Combined baseline (frequency + recency)...")

# Simple combination: normalize and add scores
combined_predictions = {}
last_used = {p: pd.Timestamp('1990-01-01') for p in range(1, 40)}

# Normalize frequency to 0-1
freq_min, freq_max = freq_counts.min(), freq_counts.max()
freq_norm = {p: (freq_counts.get(p, 0) - freq_min) / (freq_max - freq_min) for p in range(1, 40)}

for idx, row in df.iterrows():
    date = row['date']
    actual = get_actual_parts(row)

    if date >= cutoff_date:
        # Recency score (normalized days since last use)
        days_since = {p: (date - last_used[p]).days for p in range(1, 40)}
        max_days = max(days_since.values())
        recency_norm = {p: days_since[p] / max_days if max_days > 0 else 0 for p in range(1, 40)}

        # Combined score (equal weights)
        combined_scores = {p: 0.5 * freq_norm[p] + 0.5 * recency_norm[p] for p in range(1, 40)}
        ranked = sorted(combined_scores.keys(), key=lambda p: -combined_scores[p])
        combined_predictions[date] = ranked

    for p in actual:
        last_used[p] = date

combined_results = evaluate_baseline(combined_predictions, test_df, K_VALUES)
best_k_combined = max(combined_results.keys(), key=lambda k: combined_results[k]['good_or_better_rate'])
print(f"    Best K={best_k_combined}: Good-or-better = {combined_results[best_k_combined]['good_or_better_rate']:.1%}")

# ============================================================
# COMPILE RESULTS
# ============================================================
print("\n[3/6] Compiling results...")

all_results = {
    'Frequency': freq_results,
    'Recency': recency_results,
    f'Last-{best_window}-Days': lastn_results,
    'Co-occurrence': cooccur_results,
    'Combined': combined_results
}

# Create summary DataFrame
summary_rows = []
for baseline_name, results in all_results.items():
    for k, metrics in results.items():
        summary_rows.append({
            'baseline': baseline_name,
            'K': k,
            'excellent_rate': metrics['excellent_rate'],
            'good_rate': metrics['good_rate'],
            'good_or_better_rate': metrics['good_or_better_rate'],
            'unacceptable_rate': metrics['unacceptable_rate']
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_FOLDER / 'metrics.csv', index=False)
print(f"  Saved: metrics.csv")

# ============================================================
# FIND BEST BASELINE
# ============================================================
print("\n[4/6] Identifying best baseline...")

best_overall = summary_df.loc[summary_df['good_or_better_rate'].idxmax()]
print(f"\n  BEST BASELINE: {best_overall['baseline']} @ K={int(best_overall['K'])}")
print(f"    Excellent rate: {best_overall['excellent_rate']:.1%}")
print(f"    Good rate: {best_overall['good_rate']:.1%}")
print(f"    Good-or-better: {best_overall['good_or_better_rate']:.1%}")
print(f"    Unacceptable: {best_overall['unacceptable_rate']:.1%}")

# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================
print("\n[5/6] Generating visualizations...")

try:
    import matplotlib.pyplot as plt

    # Plot 1: Baseline comparison at best K
    fig, ax = plt.subplots(figsize=(12, 6))

    baselines = list(all_results.keys())
    best_k = int(best_overall['K'])

    gob_rates = [all_results[b][best_k]['good_or_better_rate'] * 100 for b in baselines]
    exc_rates = [all_results[b][best_k]['excellent_rate'] * 100 for b in baselines]

    x = np.arange(len(baselines))
    width = 0.35

    ax.bar(x - width/2, gob_rates, width, label='Good-or-Better', color='steelblue')
    ax.bar(x + width/2, exc_rates, width, label='Excellent', color='forestgreen')

    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title(f'Baseline Comparison @ K={best_k}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    for i, (g, e) in enumerate(zip(gob_rates, exc_rates)):
        ax.annotate(f'{g:.1f}%', (i - width/2, g + 1), ha='center', fontsize=9)
        ax.annotate(f'{e:.1f}%', (i + width/2, e + 1), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'baseline_comparison.png', dpi=150)
    plt.close()
    print("  Saved: baseline_comparison.png")

    # Plot 2: K sweep for best baseline
    fig, ax = plt.subplots(figsize=(10, 6))

    for baseline_name, results in all_results.items():
        ks = sorted(results.keys())
        gob = [results[k]['good_or_better_rate'] * 100 for k in ks]
        ax.plot(ks, gob, marker='o', label=baseline_name, linewidth=2)

    ax.set_xlabel('Pool Size (K)', fontsize=12)
    ax.set_ylabel('Good-or-Better Rate (%)', fontsize=12)
    ax.set_title('Performance vs Pool Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'k_sweep_analysis.png', dpi=150)
    plt.close()
    print("  Saved: k_sweep_analysis.png")

    # Plot 3: Tier breakdown for best baseline
    fig, ax = plt.subplots(figsize=(10, 6))

    best_baseline_name = best_overall['baseline']
    best_results = all_results[best_baseline_name]

    ks = sorted(best_results.keys())
    excellent = [best_results[k]['excellent_rate'] * 100 for k in ks]
    good = [best_results[k]['good_rate'] * 100 for k in ks]
    unacceptable = [best_results[k]['unacceptable_rate'] * 100 for k in ks]

    ax.bar(ks, excellent, label='Excellent (5/5)', color='forestgreen')
    ax.bar(ks, good, bottom=excellent, label='Good (4/5)', color='steelblue')
    ax.bar(ks, unacceptable, bottom=[e+g for e,g in zip(excellent, good)],
           label='Unacceptable (≤3/5)', color='coral')

    ax.set_xlabel('Pool Size (K)', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title(f'Tier Breakdown: {best_baseline_name}', fontsize=14)
    ax.legend()
    ax.set_xticks(ks)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'tier_breakdown.png', dpi=150)
    plt.close()
    print("  Saved: tier_breakdown.png")

    plots_generated = True
except ImportError:
    print("  [SKIP] matplotlib not installed")
    plots_generated = False

# ============================================================
# GENERATE REPORT
# ============================================================
print("\n[6/6] Generating report...")

# Summary table for report
summary_at_best_k = summary_df[summary_df['K'] == best_k].sort_values('good_or_better_rate', ascending=False)

report = f"""# Baseline Suite Report

**Run ID**: {RUN_ID}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Period**: {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} days)

---

## Executive Summary

**Best Baseline**: {best_overall['baseline']} @ K={int(best_overall['K'])}

| Metric | Value |
|--------|-------|
| Excellent Rate (5/5) | {best_overall['excellent_rate']:.1%} |
| Good Rate (4/5) | {best_overall['good_rate']:.1%} |
| **Good-or-Better** | **{best_overall['good_or_better_rate']:.1%}** |
| Unacceptable (≤3/5) | {best_overall['unacceptable_rate']:.1%} |

---

## Baseline Comparison @ K={best_k}

| Baseline | Excellent | Good | Good-or-Better | Unacceptable |
|----------|-----------|------|----------------|--------------|
"""

for _, row in summary_at_best_k.iterrows():
    report += f"| {row['baseline']} | {row['excellent_rate']:.1%} | {row['good_rate']:.1%} | {row['good_or_better_rate']:.1%} | {row['unacceptable_rate']:.1%} |\n"

report += f"""
---

## What These Results Mean

### Performance Interpretation

- **Good-or-Better = {best_overall['good_or_better_rate']:.1%}** means that on {best_overall['good_or_better_rate']*100:.0f}% of days, a pool of {int(best_overall['K'])} parts would contain at least 4 of the 5 actual parts needed.

- **Unacceptable = {best_overall['unacceptable_rate']:.1%}** means on {best_overall['unacceptable_rate']*100:.0f}% of days, the pool would miss 2 or more parts (potentially causing production issues).

### Baseline Insights

1. **{list(all_results.keys())[0]}**: Uses overall historical frequency. {'Strong performer.' if freq_results[best_k]['good_or_better_rate'] > 0.8 else 'Moderate performance.'}

2. **Recency**: Assumes parts not used recently are "due". {'Works well.' if recency_results[best_k]['good_or_better_rate'] > 0.8 else 'Limited predictive power on its own.'}

3. **Last-{best_window}-Days**: Recent frequency often captures short-term patterns.

4. **Co-occurrence**: Part associations provide some signal.

5. **Combined**: Mixing signals {'improves results.' if combined_results[best_k]['good_or_better_rate'] > freq_results[best_k]['good_or_better_rate'] else 'shows similar performance.'}

---

## Benchmark to Beat

Any advanced model (neural, symbolic, or hybrid) must exceed:

| Metric | Baseline Benchmark |
|--------|-------------------|
| Good-or-Better @ K={int(best_overall['K'])} | > {best_overall['good_or_better_rate']:.1%} |
| Unacceptable @ K={int(best_overall['K'])} | < {best_overall['unacceptable_rate']:.1%} |

**Advancing complexity without beating this baseline is research malpractice.**

---

## Artifacts Generated

- `config.yaml` - Run configuration
- `metrics.csv` - Full results table
- `baseline_comparison.png` - Visual comparison
- `k_sweep_analysis.png` - Performance vs K
- `tier_breakdown.png` - Tier distribution
- `baseline_report.md` - This report

---

## Recommended Next Steps

1. **Feature Engineering** - Can we improve with engineered features?
2. **Rulebook Draft** - Extract interpretable symbolic rules
3. **Neural Prototyping** - Only if baselines leave room for improvement

---

**Report generated by Dr. Synapse**
**Workflow**: baseline-suite
"""

with open(OUTPUT_FOLDER / 'baseline_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("  Saved: baseline_report.md")

# Save config
config = f"""# Baseline Suite Configuration
run_id: {RUN_ID}
timestamp: {datetime.now().isoformat()}
seed: {SEED}
k_values: {K_VALUES}
window_sizes_tested: {WINDOW_SIZES}
best_window: {best_window}
test_years: {TEST_YEARS}
train_records: {len(train_df)}
test_records: {len(test_df)}
"""
with open(OUTPUT_FOLDER / 'config.yaml', 'w', encoding='utf-8') as f:
    f.write(config)
print("  Saved: config.yaml")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("BASELINE SUITE COMPLETE")
print("=" * 70)
print(f"\nOutput folder: {OUTPUT_FOLDER}")
print(f"\n{'='*50}")
print("KEY RESULTS")
print(f"{'='*50}")
print(f"\nBest Baseline: {best_overall['baseline']} @ K={int(best_overall['K'])}")
print(f"  - Good-or-Better: {best_overall['good_or_better_rate']:.1%}")
print(f"  - Excellent (5/5): {best_overall['excellent_rate']:.1%}")
print(f"  - Unacceptable: {best_overall['unacceptable_rate']:.1%}")
print(f"\nThis is the benchmark to beat with advanced models.")
print("=" * 70)
