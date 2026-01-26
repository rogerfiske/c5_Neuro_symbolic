"""
Neural vs Baseline Comparison on Hard Parts
=============================================
Compare neural model performance vs frequency baseline
specifically on the 6 hardest-to-predict parts.

Since we don't have per-prediction neural outputs locally,
we'll analyze the hyperopt trials to infer hard-part behavior.

Author: Dr. Synapse (Neuro-Symbolic Research Agent)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'CA5_date.csv'
HYPEROPT_PATH = PROJECT_ROOT / 'outputs' / 'outputs' / 'hyperopt' / 'all_trials.json'
OUTPUT_FOLDER = PROJECT_ROOT / '_bmad-output' / 'synapse' / 'neural-vs-baseline' / 'run-001'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Hard parts identified
HARD_PARTS = [12, 8, 13, 22, 23, 39]
EASY_PARTS = [2, 6, 9, 10, 11, 15, 17, 18, 19, 25, 26, 28, 29]  # From part_analysis (recall >= 75%)
MEDIUM_PARTS = [p for p in range(1, 40) if p not in HARD_PARTS and p not in EASY_PARTS]

K_VALUES = [27, 30]
WINDOW = 30
TEST_YEARS = 2

print("=" * 70)
print("NEURAL VS BASELINE: HARD PARTS COMPARISON")
print("=" * 70)
print(f"Hard parts: {HARD_PARTS}")
print(f"Easy parts: {EASY_PARTS}")


# ============================================================
# Part 1: Load Data
# ============================================================
print("\n[1/5] Loading data...")

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

date_to_parts = {}
for idx, row in df.iterrows():
    d = row['date']
    parts = {int(row['m_1']), int(row['m_2']), int(row['m_3']),
             int(row['m_4']), int(row['m_5'])}
    date_to_parts[d] = parts

dates = sorted(date_to_parts.keys())
n_days = len(dates)

# Load hyperopt results
with open(HYPEROPT_PATH, 'r') as f:
    hyperopt_trials = json.load(f)

print(f"     Loaded {n_days} days, {len(hyperopt_trials)} hyperopt trials")


# ============================================================
# Part 2: Baseline Performance by Part Category
# ============================================================
print("\n[2/5] Computing baseline performance by part category...")

test_cutoff = dates[-1] - pd.Timedelta(days=365 * TEST_YEARS)
test_start_idx = next(i for i, d in enumerate(dates) if d > test_cutoff)
n_test_days = n_days - test_start_idx - 1

results_by_k = {}

for K in K_VALUES:
    # Track by category
    category_stats = {
        'hard': {'needed': 0, 'correct': 0, 'missed': 0},
        'medium': {'needed': 0, 'correct': 0, 'missed': 0},
        'easy': {'needed': 0, 'correct': 0, 'missed': 0}
    }

    # Track overall tier distribution
    tier_counts = {'Excellent': 0, 'Good': 0, 'Unacceptable': 0}

    # Track which category causes misses
    miss_by_category = {'hard': 0, 'medium': 0, 'easy': 0}

    for t_idx in range(test_start_idx, n_days - 1):
        current_date = dates[t_idx]
        target_date = dates[t_idx + 1]

        # Compute frequency scores
        freq = np.zeros(39)
        for back in range(1, WINDOW + 1):
            if t_idx - back >= 0:
                past_date = dates[t_idx - back]
                if past_date in date_to_parts:
                    for p in date_to_parts[past_date]:
                        freq[p - 1] += 1

        # Select top-K
        pool = set(np.argsort(-freq)[:K] + 1)
        actual = date_to_parts.get(target_date, set())

        if len(actual) != 5:
            continue

        # Count hits
        hits = len(pool & actual)
        if hits == 5:
            tier_counts['Excellent'] += 1
        elif hits == 4:
            tier_counts['Good'] += 1
        else:
            tier_counts['Unacceptable'] += 1

        # Track by category
        for p in actual:
            if p in HARD_PARTS:
                cat = 'hard'
            elif p in EASY_PARTS:
                cat = 'easy'
            else:
                cat = 'medium'

            category_stats[cat]['needed'] += 1
            if p in pool:
                category_stats[cat]['correct'] += 1
            else:
                category_stats[cat]['missed'] += 1
                miss_by_category[cat] += 1

    # Compute metrics
    total_predictions = sum(tier_counts.values())
    gob = (tier_counts['Excellent'] + tier_counts['Good']) / total_predictions * 100

    results_by_k[K] = {
        'K': K,
        'total_predictions': total_predictions,
        'gob': gob,
        'excellent_pct': tier_counts['Excellent'] / total_predictions * 100,
        'good_pct': tier_counts['Good'] / total_predictions * 100,
        'unacceptable_pct': tier_counts['Unacceptable'] / total_predictions * 100,
        'hard_recall': category_stats['hard']['correct'] / category_stats['hard']['needed'] * 100,
        'medium_recall': category_stats['medium']['correct'] / category_stats['medium']['needed'] * 100,
        'easy_recall': category_stats['easy']['correct'] / category_stats['easy']['needed'] * 100,
        'hard_misses': category_stats['hard']['missed'],
        'medium_misses': category_stats['medium']['missed'],
        'easy_misses': category_stats['easy']['missed'],
        'total_misses': sum(category_stats[c]['missed'] for c in ['hard', 'medium', 'easy']),
        'hard_miss_pct': miss_by_category['hard'] / sum(miss_by_category.values()) * 100 if sum(miss_by_category.values()) > 0 else 0,
    }

print("\n     Baseline Performance by K and Part Category:")
print("     " + "-" * 70)
print(f"     K   | GoB%   | Hard Recall | Medium Recall | Easy Recall | Hard Miss%")
print("     " + "-" * 70)
for K in K_VALUES:
    r = results_by_k[K]
    print(f"     {K:<3} | {r['gob']:.1f}% | {r['hard_recall']:.1f}%       | {r['medium_recall']:.1f}%         | {r['easy_recall']:.1f}%       | {r['hard_miss_pct']:.1f}%")


# ============================================================
# Part 3: Analyze Hyperopt Trials for Category Patterns
# ============================================================
print("\n[3/5] Analyzing hyperopt trials for category patterns...")

# We can't directly measure per-part neural performance without the predictions,
# but we can analyze which hyperparameters correlate with better overall performance
# and infer if certain configs might help hard parts

# Group trials by key parameters
param_analysis = []
for trial in hyperopt_trials:
    if trial['state'] == 'COMPLETE' and trial['value'] > 0:
        param_analysis.append({
            'trial': trial['number'],
            'value': trial['value'],
            'encoder': trial['params']['encoder_type'],
            'seq_len': trial['params']['sequence_length'],
            'pool_size': trial['params']['pool_size'],
            'num_rules': trial['params']['num_rules'],
            'embed_dim': trial['params']['embed_dim'],
            'num_layers': trial['params']['num_layers'],
            'dropout': trial['params']['dropout'],
            'use_symbolic': trial['params'].get('use_symbolic_init', False)
        })

param_df = pd.DataFrame(param_analysis)

# Best trials
best_trials = param_df.nlargest(10, 'value')

print("\n     Top 10 Hyperopt Trials:")
print("     " + "-" * 80)
print(f"     {'Trial':<6} {'GoB%':<8} {'Encoder':<12} {'SeqLen':<8} {'K':<4} {'Rules':<6} {'Layers':<7}")
print("     " + "-" * 80)
for _, row in best_trials.iterrows():
    print(f"     {int(row['trial']):<6} {row['value']:<8.1f} {row['encoder']:<12} {int(row['seq_len']):<8} "
          f"{int(row['pool_size']):<4} {int(row['num_rules']):<6} {int(row['num_layers']):<7}")

# Correlation analysis
print("\n     Parameter Correlations with Performance:")
print("     " + "-" * 50)
numeric_params = ['seq_len', 'pool_size', 'num_rules', 'embed_dim', 'num_layers', 'dropout']
for param in numeric_params:
    corr = param_df['value'].corr(param_df[param])
    direction = "+" if corr > 0 else "-"
    print(f"     {param:<12}: r = {corr:+.3f} ({direction})")


# ============================================================
# Part 4: Estimate Neural Lift on Hard Parts
# ============================================================
print("\n[4/5] Estimating neural lift on hard parts...")

# The neural model at K=30 achieves ~72% GoB vs baseline 69%
# This is +3pp overall. We need to estimate how this distributes across categories.

# Hypothesis: If neural model learns better patterns for hard parts,
# the lift should be disproportionately from improving hard part recall.

# Let's calculate what recall improvements would be needed to explain the +3pp lift

baseline_k30 = results_by_k[30]
neural_gob = 72.4  # From hyperopt
lift = neural_gob - baseline_k30['gob']

# Current misses at K=30
total_misses = baseline_k30['total_misses']
hard_misses = baseline_k30['hard_misses']
medium_misses = baseline_k30['medium_misses']
easy_misses = baseline_k30['easy_misses']

print(f"\n     Baseline @K=30 Miss Distribution:")
print(f"     Total misses: {total_misses}")
print(f"     - Hard parts:   {hard_misses} ({hard_misses/total_misses*100:.1f}%)")
print(f"     - Medium parts: {medium_misses} ({medium_misses/total_misses*100:.1f}%)")
print(f"     - Easy parts:   {easy_misses} ({easy_misses/total_misses*100:.1f}%)")

# Estimate predictions improved
# +3pp GoB on ~729 test days = ~22 more "Good" days
# Each day can improve by getting 1+ more parts right
improved_days = int(lift / 100 * baseline_k30['total_predictions'])

print(f"\n     Neural Lift Analysis:")
print(f"     Overall lift: +{lift:.1f}pp ({baseline_k30['gob']:.1f}% -> {neural_gob:.1f}%)")
print(f"     Estimated improved days: ~{improved_days}")

# Scenario analysis: Where could the lift come from?
scenarios = [
    ("All from hard parts", hard_misses, improved_days),
    ("All from medium parts", medium_misses, improved_days),
    ("Proportional to misses", total_misses, improved_days),
]

print(f"\n     Scenario Analysis - Where does +3pp come from?")
print("     " + "-" * 60)
for scenario_name, miss_pool, improvement in scenarios:
    if miss_pool > 0:
        reduction_pct = improvement / miss_pool * 100
        print(f"     {scenario_name}: Would reduce {improvement}/{miss_pool} misses ({reduction_pct:.1f}%)")


# ============================================================
# Part 5: Recommendations
# ============================================================
print("\n[5/5] Generating recommendations...")

# Key insight: Hard parts account for disproportionate misses
hard_miss_ratio = hard_misses / total_misses
hard_occurrence_ratio = len(HARD_PARTS) / 39

print(f"\n     Key Insight:")
print(f"     Hard parts are {len(HARD_PARTS)}/39 = {len(HARD_PARTS)/39*100:.1f}% of parts")
print(f"     But account for {hard_miss_ratio*100:.1f}% of misses")
print(f"     Disproportionality factor: {hard_miss_ratio / (len(HARD_PARTS)/39):.2f}x")

recommendations = []

if hard_miss_ratio > 0.25:
    recommendations.append("Hard parts are high-value targets for neural improvement")

if baseline_k30['hard_recall'] < baseline_k30['easy_recall'] - 10:
    recommendations.append(f"Recall gap: Hard ({baseline_k30['hard_recall']:.0f}%) vs Easy ({baseline_k30['easy_recall']:.0f}%)")

recommendations.append("Neural model likely provides most value on hard parts")
recommendations.append("Ensemble: Use neural specifically for hard part predictions")


# ============================================================
# Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving outputs...")

# Save baseline results
baseline_df = pd.DataFrame([results_by_k[K] for K in K_VALUES])
baseline_df.to_csv(OUTPUT_FOLDER / 'baseline_by_category.csv', index=False)

# Save hyperopt analysis
param_df.to_csv(OUTPUT_FOLDER / 'hyperopt_analysis.csv', index=False)

# Visualization
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Recall by category
    ax1 = axes[0, 0]
    categories = ['Hard', 'Medium', 'Easy']
    x = np.arange(len(categories))
    width = 0.35

    for i, K in enumerate(K_VALUES):
        r = results_by_k[K]
        recalls = [r['hard_recall'], r['medium_recall'], r['easy_recall']]
        ax1.bar(x + i*width, recalls, width, label=f'K={K}')

    ax1.set_ylabel('Recall %')
    ax1.set_title('Baseline Recall by Part Category')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5)

    # Plot 2: Miss distribution
    ax2 = axes[0, 1]
    r30 = results_by_k[30]
    miss_data = [r30['hard_misses'], r30['medium_misses'], r30['easy_misses']]
    colors = ['red', 'orange', 'green']
    ax2.pie(miss_data, labels=categories, autopct='%1.1f%%', colors=colors)
    ax2.set_title(f'Miss Distribution @K=30 (Total: {r30["total_misses"]})')

    # Plot 3: Hyperopt performance distribution
    ax3 = axes[1, 0]
    ax3.hist(param_df['value'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=baseline_k30['gob'], color='red', linestyle='--', linewidth=2, label=f'Baseline @K=30')
    ax3.axvline(x=72.4, color='green', linestyle='--', linewidth=2, label='Best Neural')
    ax3.set_xlabel('Good-or-Better %')
    ax3.set_ylabel('Trial Count')
    ax3.set_title('Hyperopt Trial Distribution')
    ax3.legend()

    # Plot 4: Expected impact
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
NEURAL VS BASELINE SUMMARY
==========================

Baseline @K=30:
  Overall GoB: {baseline_k30['gob']:.1f}%
  Hard recall: {baseline_k30['hard_recall']:.1f}%
  Easy recall: {baseline_k30['easy_recall']:.1f}%
  Gap: {baseline_k30['easy_recall'] - baseline_k30['hard_recall']:.1f}pp

Best Neural @K=30:
  Overall GoB: 72.4%
  Lift: +{lift:.1f}pp

Miss Analysis:
  Hard parts: {len(HARD_PARTS)}/39 parts ({len(HARD_PARTS)/39*100:.1f}%)
  Hard misses: {hard_miss_ratio*100:.1f}% of all misses
  Disproportionality: {hard_miss_ratio / (len(HARD_PARTS)/39):.2f}x

RECOMMENDATION:
Neural model value is likely concentrated
on hard parts. Consider ensemble:
- Baseline for easy parts
- Neural for hard parts
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'neural_vs_baseline.png', dpi=150)
    plt.close()
    print("     Saved: neural_vs_baseline.png")

except ImportError:
    print("     (matplotlib not available)")


# Generate report
report = f"""# Neural vs Baseline: Hard Parts Comparison

## Executive Summary

Analysis of baseline performance by part category reveals that
**hard parts account for disproportionate prediction failures**.

| Metric | Value |
|--------|-------|
| Hard parts | {len(HARD_PARTS)}/39 ({len(HARD_PARTS)/39*100:.1f}%) |
| Hard part miss share | {hard_miss_ratio*100:.1f}% |
| Disproportionality | {hard_miss_ratio / (len(HARD_PARTS)/39):.2f}x |

---

## Baseline Performance by Category

### @K=27

| Category | Parts | Recall | Misses |
|----------|-------|--------|--------|
| Hard | {len(HARD_PARTS)} | {results_by_k[27]['hard_recall']:.1f}% | {results_by_k[27]['hard_misses']} |
| Medium | {len(MEDIUM_PARTS)} | {results_by_k[27]['medium_recall']:.1f}% | {results_by_k[27]['medium_misses']} |
| Easy | {len(EASY_PARTS)} | {results_by_k[27]['easy_recall']:.1f}% | {results_by_k[27]['easy_misses']} |

### @K=30

| Category | Parts | Recall | Misses |
|----------|-------|--------|--------|
| Hard | {len(HARD_PARTS)} | {results_by_k[30]['hard_recall']:.1f}% | {results_by_k[30]['hard_misses']} |
| Medium | {len(MEDIUM_PARTS)} | {results_by_k[30]['medium_recall']:.1f}% | {results_by_k[30]['medium_misses']} |
| Easy | {len(EASY_PARTS)} | {results_by_k[30]['easy_recall']:.1f}% | {results_by_k[30]['easy_misses']} |

---

## Neural Model Lift Analysis

| Metric | Baseline @K=30 | Neural @K=30 | Delta |
|--------|----------------|--------------|-------|
| Overall GoB | {baseline_k30['gob']:.1f}% | 72.4% | +{lift:.1f}pp |

### Where Does +{lift:.1f}pp Come From?

The neural lift of +{lift:.1f}pp represents ~{improved_days} additional "Good" days.

**Hypothesis:** If neural model specifically improves hard part predictions,
the ensemble value is maximized by using neural for hard parts only.

---

## Recommendations

"""

for i, rec in enumerate(recommendations, 1):
    report += f"{i}. {rec}\n"

report += f"""

---

## Ensemble Strategy

```
For each prediction:
  if part in HARD_PARTS ({HARD_PARTS}):
    use neural_model_score
  else:
    use frequency_baseline_score
```

**Expected Benefit:**
- Maintain baseline performance on easy parts
- Gain neural lift on hard parts
- Net improvement potentially > {lift:.1f}pp if neural is disproportionately better on hard parts

---

## Next Steps

1. **Get per-prediction neural outputs** - Need to re-run inference to measure per-part accuracy
2. **Compare neural vs baseline recall** - Specifically for hard parts
3. **Implement confidence gating** - Use neural when confident, baseline otherwise
4. **Test ensemble** - Validate strategy on held-out data

---

*Generated by Dr. Synapse - Phase 2 Research*
"""

with open(OUTPUT_FOLDER / 'neural_vs_baseline_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nOutputs saved to: {OUTPUT_FOLDER}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nKey Finding: Hard parts ({HARD_PARTS}) account for {hard_miss_ratio*100:.1f}% of misses")
print(f"Recommendation: Neural model value is likely concentrated on hard parts")
print(f"\nNext step: Re-run neural inference to measure per-part accuracy")
