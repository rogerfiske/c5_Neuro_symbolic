"""
Part 12 Anomaly Investigation
=============================
Neural model has 0% recall on Part 12 while baseline gets 36.5%.
This script investigates why.

Author: Dr. Synapse Research Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the per-part results from Phase 2
results_path = Path("Phase 2 outputs/outputs/per_part_analysis/per_part_analysis/per_part_results.csv")
df = pd.read_csv(results_path)

# Load the original dataset
data_path = Path("data/raw/CA5_date.csv")
raw_df = pd.read_csv(data_path)
raw_df['date'] = pd.to_datetime(raw_df['date'])

print("=" * 70)
print("PART 12 ANOMALY INVESTIGATION")
print("=" * 70)

# ============================================================
# 1. Part 12 in Phase 2 Results
# ============================================================
print("\n## 1. Part 12 in Test Set Results\n")

part12 = df[df['part_id'] == 12]
print(f"Total Part 12 occurrences in test set: {len(part12)}")
print(f"Neural recall: {part12['in_neural_pool'].mean()*100:.1f}%")
print(f"Baseline recall: {part12['in_baseline_pool'].mean()*100:.1f}%")

print("\n### Neural Probability Stats for Part 12:")
print(f"  Mean prob:   {part12['neural_prob'].mean():.4f}")
print(f"  Min prob:    {part12['neural_prob'].min():.4f}")
print(f"  Max prob:    {part12['neural_prob'].max():.4f}")
print(f"  Std prob:    {part12['neural_prob'].std():.4f}")

print("\n### Neural Rank Stats for Part 12 (lower = better):")
print(f"  Mean rank:   {part12['neural_rank'].mean():.1f}")
print(f"  Min rank:    {part12['neural_rank'].min()}")
print(f"  Max rank:    {part12['neural_rank'].max()}")
print(f"  Rank <= 30 (in pool): {(part12['neural_rank'] <= 30).sum()} / {len(part12)}")

# ============================================================
# 2. Compare with Other Hard Parts
# ============================================================
print("\n## 2. Comparison with Other Hard Parts\n")

hard_parts = [12, 8, 13, 22, 23, 39]
print("| Part | Occurrences | Neural Recall | Mean Neural Prob | Mean Rank |")
print("|------|-------------|---------------|------------------|-----------|")

for part_id in hard_parts:
    part_df = df[df['part_id'] == part_id]
    print(f"| {part_id:4d} | {len(part_df):11d} | {part_df['in_neural_pool'].mean()*100:13.1f}% | "
          f"{part_df['neural_prob'].mean():16.4f} | {part_df['neural_rank'].mean():9.1f} |")

# ============================================================
# 3. Part 12 Frequency Analysis in Full Dataset
# ============================================================
print("\n## 3. Part 12 Frequency in Full Dataset\n")

# Flatten all parts
all_parts = []
for col in ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']:
    all_parts.extend(raw_df[col].tolist())

part_counts = pd.Series(all_parts).value_counts().sort_index()
total_days = len(raw_df)

print("### All Parts Frequency (sorted by count):")
print("| Part | Count | Frequency | Rank |")
print("|------|-------|-----------|------|")

sorted_counts = part_counts.sort_values(ascending=False)
for rank, (part_id, count) in enumerate(sorted_counts.items(), 1):
    freq = count / total_days * 100
    marker = " <-- PART 12" if part_id == 12 else ""
    print(f"| {part_id:4d} | {count:5d} | {freq:8.2f}% | {rank:4d} |{marker}")

print(f"\nPart 12 specifically:")
print(f"  Count: {part_counts.get(12, 0)}")
print(f"  Frequency: {part_counts.get(12, 0) / total_days * 100:.2f}%")
print(f"  Rank: {list(sorted_counts.index).index(12) + 1} of 39")

# ============================================================
# 4. Part 12 Temporal Distribution
# ============================================================
print("\n## 4. Part 12 Temporal Distribution\n")

# Find all days with Part 12
part12_days = raw_df[
    (raw_df['m_1'] == 12) |
    (raw_df['m_2'] == 12) |
    (raw_df['m_3'] == 12) |
    (raw_df['m_4'] == 12) |
    (raw_df['m_5'] == 12)
].copy()

print(f"Total days with Part 12: {len(part12_days)}")
print(f"Date range: {part12_days['date'].min()} to {part12_days['date'].max()}")

# By year
part12_days['year'] = part12_days['date'].dt.year
yearly = part12_days.groupby('year').size()
all_yearly = raw_df.groupby(raw_df['date'].dt.year).size()

print("\n### Part 12 by Year (last 10 years):")
print("| Year | Part 12 Days | Total Days | Frequency |")
print("|------|--------------|------------|-----------|")
for year in sorted(yearly.index)[-10:]:
    p12_count = yearly.get(year, 0)
    total = all_yearly.get(year, 1)
    print(f"| {year} | {p12_count:12d} | {total:10d} | {p12_count/total*100:8.2f}% |")

# ============================================================
# 5. Train/Val/Test Split Analysis
# ============================================================
print("\n## 5. Part 12 in Train/Val/Test Splits\n")

# Approximate splits based on test_years=2.0, val_years=0.5
last_date = raw_df['date'].max()
test_cutoff = last_date - pd.Timedelta(days=int(365 * 2.0))
val_cutoff = test_cutoff - pd.Timedelta(days=int(365 * 0.5))

train_df = raw_df[raw_df['date'] < val_cutoff]
val_df = raw_df[(raw_df['date'] >= val_cutoff) & (raw_df['date'] < test_cutoff)]
test_df = raw_df[raw_df['date'] >= test_cutoff]

def count_part(df, part_id):
    return ((df['m_1'] == part_id) | (df['m_2'] == part_id) |
            (df['m_3'] == part_id) | (df['m_4'] == part_id) |
            (df['m_5'] == part_id)).sum()

print("| Split | Total Days | Part 12 Days | Part 12 Freq |")
print("|-------|------------|--------------|--------------|")
for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    p12_count = count_part(split_df, 12)
    print(f"| {name:5s} | {len(split_df):10d} | {p12_count:12d} | {p12_count/len(split_df)*100:11.2f}% |")

# ============================================================
# 6. Part 12 Co-occurrence Analysis
# ============================================================
print("\n## 6. Part 12 Co-occurrence Analysis\n")

# What parts appear with Part 12?
cooccurrence = {}
for _, row in part12_days.iterrows():
    parts = [row['m_1'], row['m_2'], row['m_3'], row['m_4'], row['m_5']]
    for p in parts:
        if p != 12:
            cooccurrence[p] = cooccurrence.get(p, 0) + 1

print("### Parts that most often appear WITH Part 12:")
print("| Part | Co-occurrences | % of Part 12 Days |")
print("|------|----------------|-------------------|")
for part_id, count in sorted(cooccurrence.items(), key=lambda x: -x[1])[:10]:
    print(f"| {part_id:4d} | {count:14d} | {count/len(part12_days)*100:17.1f}% |")

# ============================================================
# 7. Gap Analysis - Time Since Last Part 12
# ============================================================
print("\n## 7. Gap Analysis - Days Between Part 12 Occurrences\n")

part12_dates = part12_days['date'].sort_values().reset_index(drop=True)
gaps = part12_dates.diff().dt.days.dropna()

print(f"Mean gap between Part 12 occurrences: {gaps.mean():.1f} days")
print(f"Median gap: {gaps.median():.1f} days")
print(f"Min gap: {gaps.min():.0f} days")
print(f"Max gap: {gaps.max():.0f} days")
print(f"Std gap: {gaps.std():.1f} days")

print("\n### Gap Distribution:")
print(f"  1-7 days:   {(gaps <= 7).sum()} ({(gaps <= 7).mean()*100:.1f}%)")
print(f"  8-14 days:  {((gaps > 7) & (gaps <= 14)).sum()} ({((gaps > 7) & (gaps <= 14)).mean()*100:.1f}%)")
print(f"  15-30 days: {((gaps > 14) & (gaps <= 30)).sum()} ({((gaps > 14) & (gaps <= 30)).mean()*100:.1f}%)")
print(f"  31+ days:   {(gaps > 30).sum()} ({(gaps > 30).mean()*100:.1f}%)")

# ============================================================
# 8. Neural Probability Comparison
# ============================================================
print("\n## 8. Neural Probability Distribution by Part\n")

print("### Mean Neural Probability by Part (sorted):")
part_probs = df.groupby('part_id')['neural_prob'].mean().sort_values()

print("| Part | Mean Prob | Category |")
print("|------|-----------|----------|")
for part_id, prob in part_probs.items():
    cat = 'HARD' if part_id in hard_parts else ('EASY' if part_id in [2, 6, 9, 10, 11, 15, 17, 18, 19, 25, 26, 28, 29] else 'MED')
    marker = " <-- PART 12" if part_id == 12 else ""
    print(f"| {part_id:4d} | {prob:.5f} | {cat:8s} |{marker}")

# ============================================================
# 9. Key Finding Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: WHY PART 12 FAILS")
print("=" * 70)

# Get Part 12's rank in neural probabilities
part12_prob = df[df['part_id'] == 12]['neural_prob'].mean()
prob_rank = (part_probs < part12_prob).sum() + 1

print(f"""
1. FREQUENCY: Part 12 is ranked {list(sorted_counts.index).index(12) + 1}/39 in overall frequency
   - This means it's not particularly rare

2. NEURAL PROBABILITY: Part 12 has mean prob {part12_prob:.4f}
   - This ranks {prob_rank}/39 among all parts
   - The model assigns it LOWER probability than most parts

3. NEURAL RANK: Part 12 averages rank {part12['neural_rank'].mean():.1f}
   - Pool size is K=30, so it's consistently OUTSIDE the pool
   - Best rank achieved: {part12['neural_rank'].min()} (still outside K=30)

4. POSSIBLE CAUSES:
   - Model may have learned Part 12 is "not predictable" from recent history
   - Part 12's occurrence pattern may not correlate with the 14-day window
   - May be a genuine edge case the Transformer architecture struggles with

5. RECOMMENDED ACTIONS:
   - Check attention weights for Part 12 predictions
   - Try longer sequence lengths (Part 12 may have longer cycles)
   - Consider special handling for Part 12 in production
""")
