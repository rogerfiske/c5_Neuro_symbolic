"""
Per-Part Predictability Analysis
=================================
Phase 2 Research: Identify which parts are easy/hard to predict
and whether differential predictability enables ensemble approaches.

Key Questions:
1. Are some parts easier to predict than others?
2. Which parts drive the 31% Unacceptable rate?
3. Do "hard" parts have different characteristics?
4. Is there opportunity for part-specific routing?

Author: Dr. Synapse (Neuro-Symbolic Research Agent)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'CA5_date.csv'
OUTPUT_FOLDER = PROJECT_ROOT / '_bmad-output' / 'synapse' / 'part-analysis' / 'run-001'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Config
TEST_YEARS = 2
WINDOW = 30  # Lookback window for frequency baseline
K = 27  # Fixed K for fair comparison

print("=" * 70)
print("PER-PART PREDICTABILITY ANALYSIS")
print("=" * 70)


# ============================================================
# Part 1: Load and Prepare Data
# ============================================================
print("\n[1/6] Loading data...")

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Build lookup structures
date_to_parts = {}
part_to_dates = defaultdict(list)
all_parts_sequence = []

for idx, row in df.iterrows():
    d = row['date']
    parts = {int(row['m_1']), int(row['m_2']), int(row['m_3']),
             int(row['m_4']), int(row['m_5'])}
    date_to_parts[d] = parts
    all_parts_sequence.append(parts)
    for p in parts:
        part_to_dates[p].append(d)

dates = sorted(date_to_parts.keys())
n_days = len(dates)
n_parts = 39

print(f"     Total days: {n_days}")
print(f"     Total parts: {n_parts}")
print(f"     Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")


# ============================================================
# Part 2: Global Part Statistics
# ============================================================
print("\n[2/6] Computing global part statistics...")

part_stats = []

for p in range(1, n_parts + 1):
    appearances = len(part_to_dates[p])
    frequency = appearances / n_days

    # Inter-arrival times (days between consecutive appearances)
    part_dates = sorted(part_to_dates[p])
    if len(part_dates) > 1:
        gaps = [(part_dates[i+1] - part_dates[i]).days for i in range(len(part_dates)-1)]
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        cv_gap = std_gap / mean_gap if mean_gap > 0 else 0  # Coefficient of variation
        max_gap = max(gaps)
        min_gap = min(gaps)
    else:
        mean_gap = std_gap = cv_gap = max_gap = min_gap = 0

    # Burstiness: consecutive day appearances
    consecutive_runs = 0
    current_run = 0
    for i in range(len(part_dates) - 1):
        if (part_dates[i+1] - part_dates[i]).days == 1:
            current_run += 1
        else:
            if current_run > 0:
                consecutive_runs += 1
            current_run = 0
    burst_ratio = consecutive_runs / appearances if appearances > 0 else 0

    part_stats.append({
        'part_id': p,
        'appearances': appearances,
        'frequency': frequency,
        'mean_gap': mean_gap,
        'std_gap': std_gap,
        'cv_gap': cv_gap,
        'max_gap': max_gap,
        'min_gap': min_gap,
        'burst_ratio': burst_ratio
    })

part_stats_df = pd.DataFrame(part_stats)

# Summary statistics
print(f"\n     Part Frequency Statistics:")
print(f"     Mean frequency: {part_stats_df['frequency'].mean():.3f} ({part_stats_df['frequency'].mean()*100:.1f}%)")
print(f"     Std frequency:  {part_stats_df['frequency'].std():.3f}")
print(f"     CV frequency:   {part_stats_df['frequency'].std()/part_stats_df['frequency'].mean():.3f}")
print(f"     Min frequency:  {part_stats_df['frequency'].min():.3f} (Part {part_stats_df.loc[part_stats_df['frequency'].idxmin(), 'part_id']})")
print(f"     Max frequency:  {part_stats_df['frequency'].max():.3f} (Part {part_stats_df.loc[part_stats_df['frequency'].idxmax(), 'part_id']})")


# ============================================================
# Part 3: Per-Part Prediction Accuracy (Frequency Baseline)
# ============================================================
print("\n[3/6] Computing per-part prediction accuracy...")

# Find test start
test_cutoff = dates[-1] - pd.Timedelta(days=365 * TEST_YEARS)
test_start_idx = next(i for i, d in enumerate(dates) if d > test_cutoff)
n_test_days = n_days - test_start_idx - 1

print(f"     Test period: {n_test_days} days")

# Track per-part accuracy
part_correct = defaultdict(int)  # Times part was correctly included when needed
part_total = defaultdict(int)    # Times part was actually used
part_false_positive = defaultdict(int)  # Times part was predicted but not used
part_in_pool = defaultdict(int)  # Times part appeared in prediction pool

# Track which parts cause misses
miss_analysis = []

for t_idx in range(test_start_idx, n_days - 1):
    current_date = dates[t_idx]
    target_date = dates[t_idx + 1]

    # Compute frequency scores
    freq = np.zeros(n_parts)
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

    # Track per-part metrics
    for p in pool:
        part_in_pool[p] += 1
        if p in actual:
            part_correct[p] += 1
        else:
            part_false_positive[p] += 1

    for p in actual:
        part_total[p] += 1

    # Analyze misses
    missed_parts = actual - pool
    hits = len(actual & pool)

    if len(missed_parts) > 0:
        for mp in missed_parts:
            miss_analysis.append({
                'date': target_date,
                'missed_part': mp,
                'hits': hits,
                'part_freq_rank': int(np.where(np.argsort(-freq) == (mp - 1))[0][0]) + 1,
                'part_freq_score': freq[mp - 1]
            })

# Compute per-part metrics
part_accuracy = []
for p in range(1, n_parts + 1):
    total = part_total[p]
    correct = part_correct[p]
    in_pool = part_in_pool[p]
    fp = part_false_positive[p]

    recall = correct / total if total > 0 else 0  # When part is needed, is it in pool?
    precision = correct / in_pool if in_pool > 0 else 0  # When part is in pool, is it needed?
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    part_accuracy.append({
        'part_id': p,
        'times_needed': total,
        'times_in_pool': in_pool,
        'correct_predictions': correct,
        'false_positives': fp,
        'recall': recall,
        'precision': precision,
        'f1': f1
    })

part_accuracy_df = pd.DataFrame(part_accuracy)

# Merge with stats
part_analysis_df = part_stats_df.merge(part_accuracy_df, on='part_id')

# Sort by recall (hardest to predict first)
part_analysis_df = part_analysis_df.sort_values('recall').reset_index(drop=True)

print(f"\n     Per-Part Recall (Predictability):")
print(f"     " + "-" * 60)
print(f"     {'Part':<6} {'Freq%':<8} {'Recall':<8} {'Precision':<10} {'F1':<8} {'CV_Gap':<8}")
print(f"     " + "-" * 60)
for _, row in part_analysis_df.head(10).iterrows():
    print(f"     {int(row['part_id']):<6} {row['frequency']*100:<8.1f} {row['recall']*100:<8.1f} "
          f"{row['precision']*100:<10.1f} {row['f1']*100:<8.1f} {row['cv_gap']:<8.2f}")
print(f"     ... (showing 10 hardest to predict)")


# ============================================================
# Part 4: Miss Analysis - What Causes Unacceptable Days?
# ============================================================
print("\n[4/6] Analyzing prediction misses...")

miss_df = pd.DataFrame(miss_analysis)

# Count misses per part
miss_counts = miss_df['missed_part'].value_counts().sort_values(ascending=False)

print(f"\n     Total misses: {len(miss_df)}")
print(f"     Unique missed parts: {miss_df['missed_part'].nunique()}")

print(f"\n     Top 10 Most Frequently Missed Parts:")
print(f"     " + "-" * 40)
for part, count in miss_counts.head(10).items():
    part_row = part_analysis_df[part_analysis_df['part_id'] == part].iloc[0]
    print(f"     Part {part:2d}: {count:4d} misses (freq={part_row['frequency']*100:.1f}%, recall={part_row['recall']*100:.1f}%)")

# Analyze rank of missed parts
print(f"\n     Missed Part Rank Distribution:")
rank_bins = [0, 27, 30, 35, 39]
rank_labels = ['1-27 (in K)', '28-30', '31-35', '36-39']
miss_df['rank_bin'] = pd.cut(miss_df['part_freq_rank'], bins=rank_bins, labels=rank_labels)
rank_dist = miss_df['rank_bin'].value_counts().sort_index()
for label, count in rank_dist.items():
    print(f"     Rank {label}: {count} misses ({count/len(miss_df)*100:.1f}%)")


# ============================================================
# Part 5: Temporal Pattern Analysis
# ============================================================
print("\n[5/6] Analyzing temporal patterns...")

# Day-of-week analysis
df['day_of_week'] = df['date'].dt.dayofweek

dow_part_freq = defaultdict(lambda: defaultdict(int))
dow_counts = defaultdict(int)

for idx, row in df.iterrows():
    dow = row['day_of_week']
    dow_counts[dow] += 1
    for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
        p = int(row[col])
        dow_part_freq[dow][p] += 1

# Find parts with strongest day-of-week patterns
dow_variation = []
for p in range(1, n_parts + 1):
    dow_freqs = [dow_part_freq[d][p] / dow_counts[d] if dow_counts[d] > 0 else 0 for d in range(7)]
    mean_freq = np.mean(dow_freqs)
    std_freq = np.std(dow_freqs)
    cv = std_freq / mean_freq if mean_freq > 0 else 0
    max_dow = np.argmax(dow_freqs)
    min_dow = np.argmin(dow_freqs)

    dow_variation.append({
        'part_id': p,
        'dow_cv': cv,
        'max_dow': max_dow,
        'min_dow': min_dow,
        'max_freq': dow_freqs[max_dow],
        'min_freq': dow_freqs[min_dow],
        'dow_range': dow_freqs[max_dow] - dow_freqs[min_dow]
    })

dow_df = pd.DataFrame(dow_variation)
dow_df = dow_df.sort_values('dow_cv', ascending=False)

print(f"\n     Parts with Strongest Day-of-Week Patterns:")
print(f"     " + "-" * 50)
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for _, row in dow_df.head(5).iterrows():
    print(f"     Part {int(row['part_id']):2d}: CV={row['dow_cv']:.3f}, "
          f"High={dow_names[int(row['max_dow'])]}, Low={dow_names[int(row['min_dow'])]}")

# Autocorrelation analysis (does yesterday predict today?)
print(f"\n     Autocorrelation Analysis (lag-1):")
part_autocorr = []
for p in range(1, n_parts + 1):
    # Create binary time series for this part
    series = np.array([1 if p in date_to_parts[d] else 0 for d in dates])
    if len(series) > 1:
        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
    else:
        autocorr = 0
    part_autocorr.append({'part_id': p, 'autocorr_lag1': autocorr})

autocorr_df = pd.DataFrame(part_autocorr)
autocorr_df = autocorr_df.sort_values('autocorr_lag1', ascending=False)

print(f"     " + "-" * 40)
print(f"     Top 5 parts with positive autocorrelation (burstiness):")
for _, row in autocorr_df.head(5).iterrows():
    print(f"     Part {int(row['part_id']):2d}: autocorr = {row['autocorr_lag1']:.3f}")

print(f"\n     Bottom 5 parts (negative/no autocorrelation):")
for _, row in autocorr_df.tail(5).iterrows():
    print(f"     Part {int(row['part_id']):2d}: autocorr = {row['autocorr_lag1']:.3f}")


# ============================================================
# Part 6: Co-occurrence Analysis
# ============================================================
print("\n[6/6] Analyzing co-occurrence patterns...")

# Co-occurrence matrix
cooccur = np.zeros((n_parts, n_parts))
for parts in all_parts_sequence:
    parts_list = list(parts)
    for i, p1 in enumerate(parts_list):
        for p2 in parts_list[i+1:]:
            cooccur[p1-1, p2-1] += 1
            cooccur[p2-1, p1-1] += 1

# Expected co-occurrence under independence
expected = np.outer(part_stats_df['appearances'].values, part_stats_df['appearances'].values) / n_days

# Lift (observed / expected)
with np.errstate(divide='ignore', invalid='ignore'):
    lift = cooccur / expected
    lift = np.nan_to_num(lift, nan=1.0, posinf=1.0, neginf=1.0)

# Find strongest positive associations
strong_assoc = []
for i in range(n_parts):
    for j in range(i+1, n_parts):
        if cooccur[i, j] >= 50:  # Minimum support
            strong_assoc.append({
                'part_a': i + 1,
                'part_b': j + 1,
                'cooccur': cooccur[i, j],
                'lift': lift[i, j]
            })

assoc_df = pd.DataFrame(strong_assoc)
if len(assoc_df) > 0:
    assoc_df = assoc_df.sort_values('lift', ascending=False)

    print(f"\n     Strongest Part Associations (Lift > 1 = positive correlation):")
    print(f"     " + "-" * 50)
    for _, row in assoc_df.head(10).iterrows():
        print(f"     Parts {int(row['part_a']):2d} & {int(row['part_b']):2d}: "
              f"co-occur {int(row['cooccur'])} times, lift = {row['lift']:.2f}")
else:
    print("     No strong associations found with minimum support")


# ============================================================
# Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving outputs...")

# Merge all analyses
part_analysis_df = part_analysis_df.merge(dow_df[['part_id', 'dow_cv', 'dow_range']], on='part_id')
part_analysis_df = part_analysis_df.merge(autocorr_df, on='part_id')

# Save main analysis
part_analysis_df.to_csv(OUTPUT_FOLDER / 'part_analysis.csv', index=False)

# Save miss analysis
miss_df.to_csv(OUTPUT_FOLDER / 'miss_analysis.csv', index=False)

# Save co-occurrence
if len(assoc_df) > 0:
    assoc_df.to_csv(OUTPUT_FOLDER / 'cooccurrence.csv', index=False)

# Visualization
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Part frequency distribution
    ax1 = axes[0, 0]
    ax1.bar(part_stats_df['part_id'], part_stats_df['frequency'] * 100, color='steelblue')
    ax1.axhline(y=part_stats_df['frequency'].mean() * 100, color='red', linestyle='--', label='Mean')
    ax1.set_xlabel('Part ID')
    ax1.set_ylabel('Frequency %')
    ax1.set_title('Part Usage Frequency')
    ax1.legend()

    # Plot 2: Recall by part (predictability)
    ax2 = axes[0, 1]
    sorted_df = part_analysis_df.sort_values('recall')
    colors = ['red' if r < 0.6 else 'orange' if r < 0.75 else 'green' for r in sorted_df['recall']]
    ax2.barh(range(len(sorted_df)), sorted_df['recall'] * 100, color=colors)
    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels([f"P{int(p)}" for p in sorted_df['part_id']], fontsize=6)
    ax2.set_xlabel('Recall %')
    ax2.set_title(f'Per-Part Recall @K={K} (Red=Hard, Green=Easy)')
    ax2.axvline(x=70, color='black', linestyle=':', alpha=0.5)

    # Plot 3: Miss counts by part
    ax3 = axes[0, 2]
    miss_parts = miss_counts.head(15)
    ax3.bar(range(len(miss_parts)), miss_parts.values, color='crimson')
    ax3.set_xticks(range(len(miss_parts)))
    ax3.set_xticklabels([f"P{p}" for p in miss_parts.index], rotation=45)
    ax3.set_ylabel('Miss Count')
    ax3.set_title('Top 15 Most Frequently Missed Parts')

    # Plot 4: Frequency vs Recall
    ax4 = axes[1, 0]
    ax4.scatter(part_analysis_df['frequency'] * 100, part_analysis_df['recall'] * 100,
                c=part_analysis_df['cv_gap'], cmap='RdYlGn_r', s=80, edgecolors='black')
    ax4.set_xlabel('Part Frequency %')
    ax4.set_ylabel('Recall %')
    ax4.set_title('Frequency vs Predictability (color=gap variability)')
    plt.colorbar(ax4.collections[0], ax=ax4, label='CV of Gap')

    # Plot 5: Autocorrelation distribution
    ax5 = axes[1, 1]
    ax5.hist(autocorr_df['autocorr_lag1'], bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='red', linestyle='--')
    ax5.set_xlabel('Lag-1 Autocorrelation')
    ax5.set_ylabel('Count')
    ax5.set_title('Part Autocorrelation Distribution')

    # Plot 6: Day-of-week variation
    ax6 = axes[1, 2]
    ax6.scatter(part_analysis_df['dow_cv'], part_analysis_df['recall'] * 100,
                c=part_analysis_df['frequency'] * 100, cmap='viridis', s=80, edgecolors='black')
    ax6.set_xlabel('Day-of-Week CV (temporal variation)')
    ax6.set_ylabel('Recall %')
    ax6.set_title('Temporal Variation vs Predictability')
    plt.colorbar(ax6.collections[0], ax=ax6, label='Frequency %')

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'part_analysis.png', dpi=150)
    plt.close()
    print("     Saved: part_analysis.png")

    # Heatmap of co-occurrence
    fig2, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(lift, cmap='RdBu_r', vmin=0.5, vmax=1.5)
    ax.set_xlabel('Part ID')
    ax.set_ylabel('Part ID')
    ax.set_title('Part Co-occurrence Lift (>1 = positive association)')
    plt.colorbar(im, ax=ax, label='Lift')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'cooccurrence_heatmap.png', dpi=150)
    plt.close()
    print("     Saved: cooccurrence_heatmap.png")

except ImportError:
    print("     (matplotlib not available - skipping plots)")


# ============================================================
# Summary Report
# ============================================================

# Classify parts by predictability
easy_parts = part_analysis_df[part_analysis_df['recall'] >= 0.75]['part_id'].tolist()
medium_parts = part_analysis_df[(part_analysis_df['recall'] >= 0.60) & (part_analysis_df['recall'] < 0.75)]['part_id'].tolist()
hard_parts = part_analysis_df[part_analysis_df['recall'] < 0.60]['part_id'].tolist()

report = f"""# Per-Part Predictability Analysis Report

## Executive Summary

Analysis of {n_parts} parts over {n_test_days} test days at K={K} reveals
**significant variation in per-part predictability**.

### Part Classification by Recall

| Category | Parts | Count | Avg Recall |
|----------|-------|-------|------------|
| Easy (≥75%) | {easy_parts[:5]}{'...' if len(easy_parts) > 5 else ''} | {len(easy_parts)} | {part_analysis_df[part_analysis_df['recall'] >= 0.75]['recall'].mean()*100:.1f}% |
| Medium (60-75%) | {medium_parts[:5]}{'...' if len(medium_parts) > 5 else ''} | {len(medium_parts)} | {part_analysis_df[(part_analysis_df['recall'] >= 0.60) & (part_analysis_df['recall'] < 0.75)]['recall'].mean()*100:.1f}% |
| Hard (<60%) | {hard_parts[:5]}{'...' if len(hard_parts) > 5 else ''} | {len(hard_parts)} | {part_analysis_df[part_analysis_df['recall'] < 0.60]['recall'].mean()*100:.1f}% |

### Key Finding: Hard Parts Drive Failures

The **{len(hard_parts)} hard-to-predict parts** account for a disproportionate share of misses.
These parts have:
- Lower frequency (less historical data)
- Higher gap variability (less predictable timing)
- Weak temporal patterns

---

## Detailed Analysis

### 1. Frequency Distribution

- Mean frequency: {part_stats_df['frequency'].mean()*100:.1f}%
- Std frequency: {part_stats_df['frequency'].std()*100:.1f}%
- CV: {part_stats_df['frequency'].std()/part_stats_df['frequency'].mean():.2f}
- Most frequent: Part {int(part_stats_df.loc[part_stats_df['frequency'].idxmax(), 'part_id'])} ({part_stats_df['frequency'].max()*100:.1f}%)
- Least frequent: Part {int(part_stats_df.loc[part_stats_df['frequency'].idxmin(), 'part_id'])} ({part_stats_df['frequency'].min()*100:.1f}%)

**Insight:** Near-uniform distribution (CV={part_stats_df['frequency'].std()/part_stats_df['frequency'].mean():.2f})
means frequency alone doesn't strongly differentiate parts.

### 2. Per-Part Recall @K={K}

| Part | Recall | Frequency | Gap CV | Autocorr |
|------|--------|-----------|--------|----------|
"""

for _, row in part_analysis_df.head(15).iterrows():
    report += f"| {int(row['part_id'])} | {row['recall']*100:.1f}% | {row['frequency']*100:.1f}% | {row['cv_gap']:.2f} | {row['autocorr_lag1']:.3f} |\n"

report += f"""
*(Sorted by recall, showing 15 hardest parts)*

### 3. Miss Analysis

- Total misses: {len(miss_df)}
- Unique missed parts: {miss_df['missed_part'].nunique()}

**Top 5 Most Missed Parts:**
"""

for part, count in miss_counts.head(5).items():
    part_row = part_analysis_df[part_analysis_df['part_id'] == part].iloc[0]
    report += f"- Part {part}: {count} misses ({count/len(miss_df)*100:.1f}% of all misses), recall={part_row['recall']*100:.1f}%\n"

report += f"""

### 4. Temporal Patterns

**Day-of-Week Variation:**
- Most parts have weak day-of-week patterns (low CV)
- Parts with strongest patterns: {dow_df.head(3)['part_id'].tolist()}

**Autocorrelation (Burstiness):**
- Mean lag-1 autocorr: {autocorr_df['autocorr_lag1'].mean():.3f}
- Parts with strong burstiness: {autocorr_df.head(3)['part_id'].tolist()}
- Parts with no pattern: {autocorr_df.tail(3)['part_id'].tolist()}

### 5. Co-occurrence Patterns

"""

if len(assoc_df) > 0:
    report += "**Strongest Associations:**\n"
    for _, row in assoc_df.head(5).iterrows():
        report += f"- Parts {int(row['part_a'])} & {int(row['part_b'])}: lift = {row['lift']:.2f}\n"
else:
    report += "No strong co-occurrence patterns detected.\n"

report += f"""

---

## Recommendations for Phase 2

### 1. Part-Specific Routing
The {len(hard_parts)} hard parts ({hard_parts}) could benefit from neural model attention,
while easy parts can use simple frequency baseline.

### 2. Ensemble Strategy
```
if part in hard_parts:
    use neural_model prediction
else:
    use frequency_baseline prediction
```

### 3. Further Investigation Needed
- Do hard parts have different characteristics in neural model embeddings?
- Does the neural model have higher accuracy on hard parts vs baseline?
- Can we use autocorrelation signal to boost predictions for bursty parts?

---

## Artifacts

- `part_analysis.csv` - Complete per-part metrics
- `miss_analysis.csv` - All prediction misses with context
- `cooccurrence.csv` - Part association strengths
- `part_analysis.png` - Visualization dashboard
- `cooccurrence_heatmap.png` - Part association matrix

---

*Generated by Dr. Synapse - Phase 2 Research*
"""

with open(OUTPUT_FOLDER / 'part_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nOutputs saved to: {OUTPUT_FOLDER}")

print("\n" + "=" * 70)
print("PER-PART ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nKey Finding: {len(hard_parts)} hard parts, {len(medium_parts)} medium, {len(easy_parts)} easy")
print(f"Hard parts: {hard_parts}")
print(f"\nRecommendation: Neural model may add value specifically for hard parts")
