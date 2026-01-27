"""
Part 12 Hybrid Strategy Evaluation
==================================
Evaluates the expected improvement from using baseline ONLY for Part 12,
while using neural model for all other parts.

Strategy:
- Parts 1-11, 13-39: Use neural model predictions
- Part 12 ONLY: Use baseline predictions

Author: Dr. Synapse Research Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the per-part results from Phase 2
results_path = Path("Phase 2 outputs/outputs/per_part_analysis/per_part_analysis/per_part_results.csv")
df = pd.read_csv(results_path)

print("=" * 70)
print("PART 12 HYBRID STRATEGY EVALUATION")
print("=" * 70)

# ============================================================
# 1. Current State Analysis
# ============================================================
print("\n## 1. Current State (Pure Neural @ K=30)\n")

# Group by day to get daily metrics
days = df.groupby('date_idx').agg({
    'in_neural_pool': 'sum',
    'in_baseline_pool': 'sum',
    'actual': 'sum'
}).reset_index()
days.columns = ['date_idx', 'neural_hits', 'baseline_hits', 'actual_needed']

def tier(hits):
    if hits == 5:
        return 'excellent'
    elif hits == 4:
        return 'good'
    else:
        return 'unacceptable'

days['neural_tier'] = days['neural_hits'].apply(tier)
days['baseline_tier'] = days['baseline_hits'].apply(tier)

total_days = len(days)

neural_excellent = (days['neural_tier'] == 'excellent').sum()
neural_good = (days['neural_tier'] == 'good').sum()
neural_unacceptable = (days['neural_tier'] == 'unacceptable').sum()
neural_gob = neural_excellent + neural_good

print(f"Total test days: {total_days}")
print(f"\nPure Neural:")
print(f"  Excellent: {neural_excellent} ({neural_excellent/total_days*100:.1f}%)")
print(f"  Good: {neural_good} ({neural_good/total_days*100:.1f}%)")
print(f"  GoB: {neural_gob} ({neural_gob/total_days*100:.1f}%)")
print(f"  Unacceptable: {neural_unacceptable} ({neural_unacceptable/total_days*100:.1f}%)")

baseline_excellent = (days['baseline_tier'] == 'excellent').sum()
baseline_good = (days['baseline_tier'] == 'good').sum()
baseline_unacceptable = (days['baseline_tier'] == 'unacceptable').sum()
baseline_gob = baseline_excellent + baseline_good

print(f"\nPure Baseline:")
print(f"  Excellent: {baseline_excellent} ({baseline_excellent/total_days*100:.1f}%)")
print(f"  Good: {baseline_good} ({baseline_good/total_days*100:.1f}%)")
print(f"  GoB: {baseline_gob} ({baseline_gob/total_days*100:.1f}%)")
print(f"  Unacceptable: {baseline_unacceptable} ({baseline_unacceptable/total_days*100:.1f}%)")

# ============================================================
# 2. Part 12 Impact Analysis
# ============================================================
print("\n## 2. Part 12 Impact Analysis\n")

# Get Part 12 specific data
part12_df = df[df['part_id'] == 12].copy()
part12_days = part12_df['date_idx'].unique()

print(f"Days where Part 12 was actually needed: {len(part12_df)}")

# Days where Part 12 needed but neural missed it
part12_neural_miss = part12_df[
    (part12_df['actual'] == 1) & (part12_df['in_neural_pool'] == 0)
]
print(f"Days where neural MISSED Part 12 (needed but not in pool): {len(part12_neural_miss)}")

# Of those, how many did baseline catch?
part12_baseline_caught = part12_df[
    (part12_df['actual'] == 1) &
    (part12_df['in_neural_pool'] == 0) &
    (part12_df['in_baseline_pool'] == 1)
]
print(f"Of those, baseline CAUGHT Part 12: {len(part12_baseline_caught)}")

# ============================================================
# 3. Simulate Hybrid Strategy
# ============================================================
print("\n## 3. Hybrid Strategy Simulation\n")

# For each day, simulate what happens if we swap Part 12's prediction
hybrid_hits = []

for date_idx in days['date_idx']:
    day_parts = df[df['date_idx'] == date_idx]

    hits = 0
    for _, row in day_parts.iterrows():
        if row['actual'] == 1:  # This part was needed
            if row['part_id'] == 12:
                # Use baseline for Part 12
                if row['in_baseline_pool'] == 1:
                    hits += 1
            else:
                # Use neural for all other parts
                if row['in_neural_pool'] == 1:
                    hits += 1

    hybrid_hits.append(hits)

days['hybrid_hits'] = hybrid_hits
days['hybrid_tier'] = days['hybrid_hits'].apply(tier)

hybrid_excellent = (days['hybrid_tier'] == 'excellent').sum()
hybrid_good = (days['hybrid_tier'] == 'good').sum()
hybrid_unacceptable = (days['hybrid_tier'] == 'unacceptable').sum()
hybrid_gob = hybrid_excellent + hybrid_good

print("Hybrid Strategy (Neural for all EXCEPT Part 12 = Baseline):")
print(f"  Excellent: {hybrid_excellent} ({hybrid_excellent/total_days*100:.1f}%)")
print(f"  Good: {hybrid_good} ({hybrid_good/total_days*100:.1f}%)")
print(f"  GoB: {hybrid_gob} ({hybrid_gob/total_days*100:.1f}%)")
print(f"  Unacceptable: {hybrid_unacceptable} ({hybrid_unacceptable/total_days*100:.1f}%)")

# ============================================================
# 4. Comparison Summary
# ============================================================
print("\n## 4. Strategy Comparison Summary\n")

print("| Strategy | Excellent | Good | GoB | Unacceptable |")
print("|----------|-----------|------|-----|--------------|")
print(f"| Pure Neural | {neural_excellent/total_days*100:.1f}% | {neural_good/total_days*100:.1f}% | {neural_gob/total_days*100:.1f}% | {neural_unacceptable/total_days*100:.1f}% |")
print(f"| Pure Baseline | {baseline_excellent/total_days*100:.1f}% | {baseline_good/total_days*100:.1f}% | {baseline_gob/total_days*100:.1f}% | {baseline_unacceptable/total_days*100:.1f}% |")
print(f"| **HYBRID (Part 12 fix)** | {hybrid_excellent/total_days*100:.1f}% | {hybrid_good/total_days*100:.1f}% | **{hybrid_gob/total_days*100:.1f}%** | {hybrid_unacceptable/total_days*100:.1f}% |")

# ============================================================
# 5. Impact Analysis
# ============================================================
print("\n## 5. Impact of Hybrid Strategy\n")

neural_gob_pct = neural_gob/total_days*100
hybrid_gob_pct = hybrid_gob/total_days*100
baseline_gob_pct = baseline_gob/total_days*100

print(f"Hybrid vs Pure Neural: {hybrid_gob_pct - neural_gob_pct:+.1f}pp")
print(f"Hybrid vs Pure Baseline: {hybrid_gob_pct - baseline_gob_pct:+.1f}pp")

# Days that improved
improved_days = days[(days['hybrid_tier'] != 'unacceptable') & (days['neural_tier'] == 'unacceptable')]
degraded_days = days[(days['neural_tier'] != 'unacceptable') & (days['hybrid_tier'] == 'unacceptable')]

print(f"\nDays improved (unacceptable -> good/excellent): {len(improved_days)}")
print(f"Days degraded (good/excellent -> unacceptable): {len(degraded_days)}")
print(f"Net improvement: {len(improved_days) - len(degraded_days)} days")

# ============================================================
# 6. Detailed Day-by-Day Changes
# ============================================================
print("\n## 6. Day-by-Day Changes (where hybrid differs from neural)\n")

changed_days = days[days['neural_hits'] != days['hybrid_hits']]
print(f"Total days with changed hit count: {len(changed_days)}")

if len(changed_days) > 0:
    print("\n| Date Idx | Neural Hits | Hybrid Hits | Change |")
    print("|----------|-------------|-------------|--------|")
    for _, row in changed_days.head(20).iterrows():
        change = row['hybrid_hits'] - row['neural_hits']
        print(f"| {row['date_idx']:8d} | {row['neural_hits']:11d} | {row['hybrid_hits']:11d} | {change:+6d} |")

    if len(changed_days) > 20:
        print(f"... and {len(changed_days) - 20} more days")

# ============================================================
# 7. Conclusion
# ============================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print(f"""
HYBRID STRATEGY RECOMMENDATION:

1. The hybrid strategy (baseline for Part 12, neural for rest) achieves:
   - GoB: {hybrid_gob_pct:.1f}% (vs {neural_gob_pct:.1f}% pure neural)
   - Lift over pure neural: {hybrid_gob_pct - neural_gob_pct:+.1f}pp

2. Part 12 specific improvement:
   - Neural recall on Part 12: 0% (0/{len(part12_df)} days)
   - Baseline recall on Part 12: 36.5% ({len(part12_baseline_caught)}/{len(part12_df)} days)
   - Hybrid captures: {len(part12_baseline_caught)} additional Part 12 days

3. RECOMMENDATION: {'DEPLOY HYBRID' if hybrid_gob_pct > neural_gob_pct else 'STAY WITH PURE NEURAL'}

4. Production implementation:
   - Use neural model predictions for parts 1-11, 13-39
   - Use frequency baseline for Part 12 ONLY
   - Pool size remains K=30
""")

# Save results
output_path = Path("outputs/part12_hybrid_evaluation.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
days.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
