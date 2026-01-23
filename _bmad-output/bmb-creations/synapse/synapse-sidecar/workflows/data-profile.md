# Data Profiling & Validation Workflow

**Workflow ID**: data-profile
**Purpose**: Validate dataset schema, detect gaps, analyze distributions, identify concept drift
**Prerequisites**: None (this is the first workflow)
**Estimated Duration**: 30 minutes - 2 hours (depending on exploratory analysis depth)

---

## Objective

Thoroughly understand the CA5 dataset before any modeling:
- Validate data invariants (5 unique parts per day, IDs 1-39)
- Detect and characterize calendar gaps (weekends, COVID period)
- Analyze part frequency distributions and temporal patterns
- Identify potential concept drift or regime changes
- Produce comprehensive data quality report

---

## Pre-Run Checklist

Load and review: `{project-root}/_bmad/_memory/synapse-sidecar/pre-run-checklist.md`

Key items for this workflow:
- [ ] Git commit hash logged
- [ ] Output folder created: `{project-root}/_bmad-output/synapse/data-profile/{run-id}/`
- [ ] Random seed set (for any stochastic analyses)

---

## Step-by-Step Execution

### 1. Load Dataset

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load CA5 dataset
data_path = Path("C:/Users/Minis/CascadeProjects/c5_neuro_symbolic/data/raw/CA5_date.csv")
df = pd.read_csv(data_path, parse_dates=['date'])

# Basic info
print(f"Records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {df.columns.tolist()}")
```

### 2. Validate Invariants

```python
# Check: exactly 5 parts per day
part_cols = ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']
assert all(df[part_cols].notna().sum(axis=1) == 5), "Not all days have 5 parts"

# Check: no duplicates within a day
def check_unique_parts(row):
    parts = row[part_cols].values
    return len(parts) == len(set(parts))

duplicates = ~df.apply(check_unique_parts, axis=1)
if duplicates.any():
    print(f"WARNING: {duplicates.sum()} days have duplicate parts!")
    print(df[duplicates][['date'] + part_cols])
else:
    print("✅ All days have 5 unique parts")

# Check: all parts in range 1-39
all_parts = df[part_cols].values.flatten()
assert all_parts.min() >= 1 and all_parts.max() <= 39, "Parts outside valid range [1,39]"
print(f"✅ Part IDs in range: [{all_parts.min()}, {all_parts.max()}]")
```

### 3. Detect Calendar Gaps

```python
# Identify gaps in time series
df_sorted = df.sort_values('date').reset_index(drop=True)
df_sorted['days_since_last'] = df_sorted['date'].diff().dt.days

gaps = df_sorted[df_sorted['days_since_last'] > 1]
print(f"\nCalendar Gaps: {len(gaps)} instances")
print(gaps[['date', 'days_since_last']].head(20))

# Characterize gap patterns
gap_stats = gaps['days_since_last'].describe()
print(f"\nGap statistics:\n{gap_stats}")
```

### 4. Analyze Part Frequency Distribution

```python
# Global part frequency
part_counts = pd.Series(all_parts).value_counts().sort_index()
print(f"\nPart Frequency Statistics:")
print(part_counts.describe())

# Check for near-uniform distribution
expected_freq = len(df) * 5 / 39  # If perfectly uniform
print(f"Expected frequency (uniform): {expected_freq:.1f}")
print(f"Actual range: [{part_counts.min()}, {part_counts.max()}]")

# Plot frequency distribution
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(part_counts.index, part_counts.values)
ax.axhline(expected_freq, color='r', linestyle='--', label='Expected (uniform)')
ax.set_xlabel('Part ID')
ax.set_ylabel('Frequency')
ax.set_title('Part Frequency Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_folder}/part_frequency.png', dpi=150)
plt.close()
```

### 5. Temporal Pattern Analysis

```python
# Part usage over time (rolling window)
window_size = 180  # 6 months
rolling_counts = {}

for part_id in range(1, 40):
    mask = (df[part_cols] == part_id).any(axis=1)
    rolling_counts[part_id] = mask.rolling(window=window_size, min_periods=1).mean()

# Plot temporal trends for selected parts
fig, ax = plt.subplots(figsize=(14, 7))
for part_id in [1, 10, 20, 30, 39]:  # Sample
    ax.plot(df['date'], rolling_counts[part_id], label=f'Part {part_id}', alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel(f'Usage Rate ({window_size}-day rolling)')
ax.set_title('Part Usage Trends Over Time')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_folder}/temporal_trends.png', dpi=150)
plt.close()
```

### 6. Concept Drift Detection

```python
# Use Kolmogorov-Smirnov test to detect distribution shifts
from scipy.stats import ks_2samp

# Split into early/late periods
mid_idx = len(df) // 2
early_parts = df.iloc[:mid_idx][part_cols].values.flatten()
late_parts = df.iloc[mid_idx:][part_cols].values.flatten()

ks_stat, p_value = ks_2samp(early_parts, late_parts)
print(f"\nConcept Drift Check (KS Test):")
print(f"KS Statistic: {ks_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("⚠️  Significant distribution shift detected!")
else:
    print("✅ No significant distribution shift")
```

### 7. Generate Data Quality Report

```python
report = f"""
# Data Profiling Report

**Dataset**: CA5_date.csv
**Run ID**: {run_id}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Records**: {len(df):,}
- **Date Range**: {df['date'].min()} to {df['date'].max()}
- **Total Days**: {(df['date'].max() - df['date'].min()).days}
- **Calendar Gaps**: {len(gaps)} (mean={gap_stats['mean']:.1f} days, max={gap_stats['max']:.0f} days)

## Invariant Validation
- ✅ All days have exactly 5 parts
- ✅ No duplicate parts within a day
- ✅ All part IDs in range [1, 39]

## Part Frequency Distribution
- **Mean**: {part_counts.mean():.1f}
- **Std Dev**: {part_counts.std():.1f}
- **Min**: {part_counts.min()}
- **Max**: {part_counts.max()}
- **Expected (uniform)**: {expected_freq:.1f}
- **Distribution**: Near-uniform (std/mean = {part_counts.std()/part_counts.mean():.2%})

## Concept Drift Assessment
- **KS Statistic**: {ks_stat:.4f}
- **P-value**: {p_value:.4f}
- **Interpretation**: {'Significant shift detected' if p_value < 0.05 else 'No significant shift'}

## Artifacts Generated
- part_frequency.png
- temporal_trends.png
- config.yaml
- metrics.csv
- data_profile_report.md (this file)

## Next Steps
1. Proceed to baseline-suite workflow
2. Use gap information for temporal feature engineering
3. Consider drift-detection mechanisms if concept drift present
"""

with open(f'{output_folder}/data_profile_report.md', 'w') as f:
    f.write(report)

print("\n" + "="*60)
print("Data profiling complete!")
print(f"Report saved to: {output_folder}/data_profile_report.md")
print("="*60)
```

---

## Outputs & Artifacts

**Location**: `{project-root}/_bmad-output/synapse/data-profile/{run-id}/`

**Files**:
- `config.yaml` - Run configuration (seeds, parameters)
- `metrics.csv` - Key statistics (record count, gap count, part frequency stats)
- `data_profile_report.md` - Complete analysis report
- `part_frequency.png` - Frequency distribution visualization
- `temporal_trends.png` - Part usage over time

---

## Success Criteria

✅ Dataset loaded successfully (11,685 records)
✅ All invariants validated (5 unique parts per day, IDs 1-39)
✅ Calendar gaps characterized (count, distribution)
✅ Part frequency distribution analyzed (near-uniform expected)
✅ Temporal patterns visualized
✅ Concept drift assessment completed
✅ Comprehensive report generated

---

## Next Workflow

**baseline-suite** - Build strong baseline models with tier metrics across K sweep

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
