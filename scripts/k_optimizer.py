"""
K-Optimizer: Pool Size Optimization
====================================
Determines optimal K under tiered service constraints.

Two analysis approaches:
1. Hyperopt trial analysis (deep learning results)
2. Local K-sweep (hybrid heuristics)

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
OUTPUT_FOLDER = PROJECT_ROOT / '_bmad-output' / 'synapse' / 'k-optimizer' / 'run-001'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Config
K_RANGE = range(20, 31)  # K values to evaluate
TEST_YEARS = 2
WINDOW = 30

print("=" * 60)
print("K-OPTIMIZER: POOL SIZE OPTIMIZATION")
print("=" * 60)


# ============================================================
# Part 1: Analyze Hyperopt Trials
# ============================================================
print("\n[1/4] Analyzing hyperopt trials...")

with open(HYPEROPT_PATH, 'r') as f:
    trials = json.load(f)

# Group trials by pool_size
k_to_results = defaultdict(list)
for trial in trials:
    if trial['state'] == 'COMPLETE' and trial['value'] > 0:
        k = trial['params']['pool_size']
        k_to_results[k].append({
            'trial': trial['number'],
            'value': trial['value'],
            'encoder': trial['params']['encoder_type'],
            'seq_len': trial['params']['sequence_length']
        })

print(f"     Found {len(trials)} trials across K values: {sorted(k_to_results.keys())}")

# Compute statistics per K
hyperopt_stats = []
for k in sorted(k_to_results.keys()):
    results = k_to_results[k]
    values = [r['value'] for r in results]
    best_trial = max(results, key=lambda x: x['value'])
    hyperopt_stats.append({
        'K': k,
        'n_trials': len(results),
        'mean_gob': np.mean(values),
        'std_gob': np.std(values),
        'max_gob': np.max(values),
        'min_gob': np.min(values),
        'best_trial': best_trial['trial'],
        'best_encoder': best_trial['encoder']
    })

hyperopt_df = pd.DataFrame(hyperopt_stats)
print("\n     Hyperopt K Analysis (Deep Learning):")
print("     " + "-" * 50)
for _, row in hyperopt_df.iterrows():
    print(f"     K={row['K']:2d}: max={row['max_gob']:.1f}%, mean={row['mean_gob']:.1f}% (n={row['n_trials']})")


# ============================================================
# Part 2: Local K-Sweep (Hybrid Heuristics)
# ============================================================
print("\n[2/4] Running local K-sweep...")

# Load data
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Build lookup
date_to_parts = {}
for idx, row in df.iterrows():
    d = row['date']
    parts = {int(row['m_1']), int(row['m_2']), int(row['m_3']),
             int(row['m_4']), int(row['m_5'])}
    date_to_parts[d] = parts

dates = sorted(date_to_parts.keys())

# Find test start
test_cutoff = dates[-1] - pd.Timedelta(days=365 * TEST_YEARS)
test_start_idx = next(i for i, d in enumerate(dates) if d > test_cutoff)


def compute_frequency_scores(t_idx):
    """Compute frequency scores for prediction at t_idx."""
    freq = np.zeros(39)
    for back in range(1, WINDOW + 1):
        if t_idx - back >= 0:
            past_date = dates[t_idx - back]
            if past_date in date_to_parts:
                for p in date_to_parts[past_date]:
                    freq[p - 1] += 1
    return freq


def evaluate_k(k):
    """Evaluate a specific K value using frequency baseline."""
    results = {'excellent': 0, 'good': 0, 'unacceptable': 0}
    jaccards = []
    prev_pool = None

    for t_idx in range(test_start_idx, len(dates) - 1):
        current_date = dates[t_idx]
        target_date = dates[t_idx + 1]

        freq = compute_frequency_scores(t_idx)
        pool = set(np.argsort(-freq)[:k] + 1)

        actual = date_to_parts.get(target_date, set())
        if len(actual) == 5:
            hits = len(pool & actual)
            if hits == 5:
                results['excellent'] += 1
            elif hits == 4:
                results['good'] += 1
            else:
                results['unacceptable'] += 1

            if prev_pool:
                jaccard = len(pool & prev_pool) / len(pool | prev_pool)
                jaccards.append(jaccard)

        prev_pool = pool

    total = sum(results.values())
    return {
        'K': k,
        'excellent_pct': results['excellent'] / total * 100,
        'good_pct': results['good'] / total * 100,
        'unacceptable_pct': results['unacceptable'] / total * 100,
        'good_or_better': (results['excellent'] + results['good']) / total * 100,
        'avg_jaccard': np.mean(jaccards) if jaccards else 0,
        'n_predictions': total
    }


# Run K-sweep
local_results = []
print("     " + "-" * 50)
for k in K_RANGE:
    result = evaluate_k(k)
    local_results.append(result)
    print(f"     K={k:2d}: GoB={result['good_or_better']:.1f}%, "
          f"Unacceptable={result['unacceptable_pct']:.1f}%, "
          f"Jaccard={result['avg_jaccard']:.2f}")

local_df = pd.DataFrame(local_results)


# ============================================================
# Part 3: Cost Analysis & Optimization
# ============================================================
print("\n[3/4] Running cost optimization...")

# Cost parameters (configurable)
COST_PARAMS = {
    'inventory_per_part': 10,      # Cost per part in pool
    'excellent_penalty': 0,        # 5/5 hits
    'good_penalty': 50,            # 4/5 hits (expedite 1)
    'unacceptable_penalty': 500,   # <=3/5 hits (major disruption)
    'stability_penalty': 5         # Per percentage point of churn
}


def compute_cost(row, deep_learning_bonus=0):
    """Compute expected cost for a given K configuration."""
    k = row['K']

    # Inventory cost
    inventory_cost = COST_PARAMS['inventory_per_part'] * k

    # Service cost (expected)
    excellent_rate = row['excellent_pct'] / 100
    good_rate = row['good_pct'] / 100
    unacceptable_rate = row['unacceptable_pct'] / 100

    service_cost = (excellent_rate * COST_PARAMS['excellent_penalty'] +
                    good_rate * COST_PARAMS['good_penalty'] +
                    unacceptable_rate * COST_PARAMS['unacceptable_penalty'])

    # Stability cost
    churn_rate = 1 - row['avg_jaccard']
    stability_cost = COST_PARAMS['stability_penalty'] * churn_rate * 100

    # Deep learning improvement (if using neural model)
    if deep_learning_bonus > 0:
        service_cost *= (1 - deep_learning_bonus)

    total_cost = inventory_cost + service_cost + stability_cost

    return {
        'inventory_cost': inventory_cost,
        'service_cost': service_cost,
        'stability_cost': stability_cost,
        'total_cost': total_cost
    }


# Compute costs for local results
for i, row in local_df.iterrows():
    costs = compute_cost(row)
    for k, v in costs.items():
        local_df.loc[i, k] = v

print("\n     Cost Analysis (Local Heuristics):")
print("     " + "-" * 50)
for _, row in local_df.iterrows():
    print(f"     K={int(row['K']):2d}: Total=${row['total_cost']:.0f} "
          f"(inv=${row['inventory_cost']:.0f}, svc=${row['service_cost']:.0f}, stab=${row['stability_cost']:.0f})")


# ============================================================
# Part 4: Determine Optimal K
# ============================================================
print("\n[4/4] Determining optimal K...")

# Find optimal by different criteria
optimal_by_cost = local_df.loc[local_df['total_cost'].idxmin()]
optimal_by_gob = local_df.loc[local_df['good_or_better'].idxmax()]
optimal_by_unacceptable = local_df.loc[local_df['unacceptable_pct'].idxmin()]

# Constraint-based: smallest K with GoB >= 50% and Unacceptable <= 50%
constrained_df = local_df[(local_df['good_or_better'] >= 50) &
                          (local_df['unacceptable_pct'] <= 50)]
if len(constrained_df) > 0:
    optimal_constrained = constrained_df.loc[constrained_df['K'].idxmin()]
else:
    optimal_constrained = optimal_by_cost

# Deep learning optimal (from hyperopt)
dl_optimal_k = hyperopt_df.loc[hyperopt_df['max_gob'].idxmax(), 'K']
dl_optimal_gob = hyperopt_df.loc[hyperopt_df['max_gob'].idxmax(), 'max_gob']

print("\n     OPTIMIZATION RESULTS")
print("     " + "=" * 50)
print(f"\n     Local Heuristics (Frequency Baseline):")
print(f"     - Lowest cost:        K={int(optimal_by_cost['K'])} (${optimal_by_cost['total_cost']:.0f})")
print(f"     - Highest GoB:        K={int(optimal_by_gob['K'])} ({optimal_by_gob['good_or_better']:.1f}%)")
print(f"     - Lowest Unacceptable: K={int(optimal_by_unacceptable['K'])} ({optimal_by_unacceptable['unacceptable_pct']:.1f}%)")
print(f"     - Constrained optimal: K={int(optimal_constrained['K'])}")

print(f"\n     Deep Learning (RunPod Hyperopt):")
print(f"     - Best performance:   K={dl_optimal_k} ({dl_optimal_gob:.1f}% GoB)")

# Final recommendation
print("\n     " + "=" * 50)
print("     RECOMMENDATION")
print("     " + "=" * 50)

# Decision logic
if dl_optimal_gob > 60:  # Strong DL performance
    recommended_k = dl_optimal_k
    rationale = "Deep learning achieves significant improvement over baseline"
    approach = "Deploy neural model with K=30"
else:
    recommended_k = int(optimal_constrained['K'])
    rationale = "Baseline performance is near theoretical maximum"
    approach = "Use frequency baseline with optimized K"

print(f"\n     Recommended K: {recommended_k}")
print(f"     Rationale: {rationale}")
print(f"     Approach: {approach}")


# ============================================================
# Save Outputs
# ============================================================
print("\n" + "=" * 60)
print("Saving outputs...")

# Save metrics
local_df.to_csv(OUTPUT_FOLDER / 'metrics.csv', index=False)
hyperopt_df.to_csv(OUTPUT_FOLDER / 'hyperopt_k_analysis.csv', index=False)

# Save config
config = {
    'k_range': list(K_RANGE),
    'test_years': TEST_YEARS,
    'window': WINDOW,
    'cost_params': COST_PARAMS,
    'optimal_k_by_cost': int(optimal_by_cost['K']),
    'optimal_k_by_gob': int(optimal_by_gob['K']),
    'optimal_k_constrained': int(optimal_constrained['K']),
    'dl_optimal_k': int(dl_optimal_k),
    'recommended_k': int(recommended_k)
}

import yaml
with open(OUTPUT_FOLDER / 'config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Save optimal K summary
optimal_summary = {
    'recommended_k': int(recommended_k),
    'rationale': rationale,
    'approach': approach,
    'local_analysis': {
        'optimal_by_cost': {'k': int(optimal_by_cost['K']), 'cost': float(optimal_by_cost['total_cost'])},
        'optimal_by_gob': {'k': int(optimal_by_gob['K']), 'gob': float(optimal_by_gob['good_or_better'])},
        'optimal_constrained': {'k': int(optimal_constrained['K'])}
    },
    'deep_learning': {
        'optimal_k': int(dl_optimal_k),
        'max_gob': float(dl_optimal_gob)
    }
}

with open(OUTPUT_FOLDER / 'optimal_k_summary.yaml', 'w') as f:
    yaml.dump(optimal_summary, f, default_flow_style=False)

# Create visualization
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Good-or-Better vs K
    ax1 = axes[0, 0]
    ax1.plot(local_df['K'], local_df['good_or_better'], 'b-o', label='Local Heuristics', linewidth=2)

    # Add hyperopt points
    for _, row in hyperopt_df.iterrows():
        ax1.scatter(row['K'], row['max_gob'], color='purple', s=100, zorder=5,
                   alpha=0.7, marker='*')
    ax1.scatter([], [], color='purple', s=100, marker='*', label='DL Best (per K)')

    ax1.axhline(y=53.1, color='red', linestyle='--', alpha=0.7, label='Baseline @K=27')
    ax1.set_xlabel('Pool Size (K)')
    ax1.set_ylabel('Good-or-Better %')
    ax1.set_title('Performance vs Pool Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tier Breakdown
    ax2 = axes[0, 1]
    ax2.stackplot(local_df['K'],
                  local_df['excellent_pct'],
                  local_df['good_pct'],
                  local_df['unacceptable_pct'],
                  labels=['Excellent (5/5)', 'Good (4/5)', 'Unacceptable'],
                  colors=['#2ecc71', '#3498db', '#e74c3c'],
                  alpha=0.8)
    ax2.set_xlabel('Pool Size (K)')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Tier Distribution vs K')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cost Analysis
    ax3 = axes[1, 0]
    ax3.plot(local_df['K'], local_df['total_cost'], 'g-o', label='Total Cost', linewidth=2)
    ax3.plot(local_df['K'], local_df['inventory_cost'], 'b--', label='Inventory', alpha=0.7)
    ax3.plot(local_df['K'], local_df['service_cost'], 'r--', label='Service', alpha=0.7)
    ax3.axvline(x=optimal_by_cost['K'], color='green', linestyle=':', alpha=0.7, label=f'Optimal K={int(optimal_by_cost["K"])}')
    ax3.set_xlabel('Pool Size (K)')
    ax3.set_ylabel('Expected Cost ($)')
    ax3.set_title('Cost Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Stability
    ax4 = axes[1, 1]
    ax4.plot(local_df['K'], local_df['avg_jaccard'], 'purple', marker='o', linewidth=2)
    ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
    ax4.set_xlabel('Pool Size (K)')
    ax4.set_ylabel('Jaccard Similarity')
    ax4.set_title('Pool Stability vs K')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'k_optimization.png', dpi=150)
    plt.close()
    print("     Saved: k_optimization.png")

    # Pareto frontier
    fig2, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(local_df['K'], local_df['good_or_better'],
                        c=local_df['total_cost'], cmap='RdYlGn_r', s=200,
                        edgecolors='black', linewidth=1)

    # Highlight recommended
    rec_row = local_df[local_df['K'] == recommended_k].iloc[0]
    ax.scatter(recommended_k, rec_row['good_or_better'],
              color='gold', s=400, marker='*', edgecolors='black',
              linewidth=2, zorder=10, label=f'Recommended K={recommended_k}')

    plt.colorbar(scatter, label='Total Cost ($)')
    ax.set_xlabel('Pool Size (K)')
    ax.set_ylabel('Good-or-Better %')
    ax.set_title('K Optimization: Performance vs Cost Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'pareto_frontier.png', dpi=150)
    plt.close()
    print("     Saved: pareto_frontier.png")

except ImportError:
    print("     (matplotlib not available - skipping plots)")

# Generate report
report = f"""# K-Optimizer Report

## Executive Summary

**Recommended Pool Size: K = {recommended_k}**

{rationale}

---

## Analysis Overview

### 1. Hyperopt Trial Analysis (Deep Learning)

The 50-trial hyperopt on RunPod H200 explored K values from 20-30:

| K | Trials | Max GoB | Mean GoB | Best Trial |
|---|--------|---------|----------|------------|
"""

for _, row in hyperopt_df.iterrows():
    report += f"| {row['K']} | {row['n_trials']} | {row['max_gob']:.1f}% | {row['mean_gob']:.1f}% | #{row['best_trial']} |\n"

report += f"""

**Key Finding:** Deep learning achieves **{dl_optimal_gob:.1f}%** Good-or-Better at K={dl_optimal_k}.

### 2. Local K-Sweep (Frequency Baseline)

Evaluated K from 20-30 using last-30-days frequency heuristic:

| K | Good-or-Better | Unacceptable | Stability | Total Cost |
|---|----------------|--------------|-----------|------------|
"""

for _, row in local_df.iterrows():
    report += f"| {int(row['K'])} | {row['good_or_better']:.1f}% | {row['unacceptable_pct']:.1f}% | {row['avg_jaccard']:.2f} | ${row['total_cost']:.0f} |\n"

report += f"""

### 3. Cost Optimization

**Cost Parameters:**
- Inventory cost: ${COST_PARAMS['inventory_per_part']}/part
- Good penalty (4/5): ${COST_PARAMS['good_penalty']}
- Unacceptable penalty (<=3/5): ${COST_PARAMS['unacceptable_penalty']}
- Stability penalty: ${COST_PARAMS['stability_penalty']}/% churn

**Optimal K by Criterion:**
- Lowest total cost: K={int(optimal_by_cost['K'])} (${optimal_by_cost['total_cost']:.0f})
- Highest Good-or-Better: K={int(optimal_by_gob['K'])} ({optimal_by_gob['good_or_better']:.1f}%)
- Lowest Unacceptable: K={int(optimal_by_unacceptable['K'])} ({optimal_by_unacceptable['unacceptable_pct']:.1f}%)

---

## Recommendation

### Final Decision: K = {recommended_k}

**Rationale:** {rationale}

**Implementation Approach:** {approach}

### Performance Summary

| Approach | K | Good-or-Better | vs Baseline |
|----------|---|----------------|-------------|
| Frequency Baseline | 27 | 53.1% | - |
| Local Heuristics | {int(optimal_by_gob['K'])} | {optimal_by_gob['good_or_better']:.1f}% | {optimal_by_gob['good_or_better'] - 53.1:+.1f}pp |
| Deep Learning | {dl_optimal_k} | {dl_optimal_gob:.1f}% | {dl_optimal_gob - 53.1:+.1f}pp |

---

## Artifacts

- `config.yaml` - Configuration parameters
- `metrics.csv` - Local K-sweep results
- `hyperopt_k_analysis.csv` - Deep learning trial analysis
- `optimal_k_summary.yaml` - Final recommendation
- `k_optimization.png` - Visualization
- `pareto_frontier.png` - Cost-performance tradeoff

---

*Generated by Dr. Synapse - K-Optimizer Workflow*
"""

with open(OUTPUT_FOLDER / 'k_optimization_report.md', 'w') as f:
    f.write(report)

print(f"\nOutputs saved to: {OUTPUT_FOLDER}")
print("\n" + "=" * 60)
print("K-OPTIMIZER COMPLETE")
print("=" * 60)
print(f"\nRecommended K: {recommended_k}")
print(f"Approach: {approach}")
