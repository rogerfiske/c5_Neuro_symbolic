"""
Data Profiling Script for CA5 Dataset
======================================
Run ID: run-001
Purpose: Understand the dataset before any modeling

What this script does (in plain English):
1. Loads the CA5 dataset (your historical parts data)
2. Checks that the data follows expected rules (5 parts per day, IDs 1-39)
3. Finds gaps in the calendar (missing days)
4. Counts how often each part is used (should be roughly equal)
5. Looks for changes in patterns over time
6. Creates a report with charts

To run: python scripts/data_profile.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RUN_ID = "run-001"
SEED = 42
np.random.seed(SEED)

# Paths
PROJECT_ROOT = Path("C:/Users/Minis/CascadeProjects/c5_neuro_symbolic")
DATA_PATH = PROJECT_ROOT / "data/raw/CA5_date.csv"
OUTPUT_FOLDER = PROJECT_ROOT / f"_bmad-output/synapse/data-profile/{RUN_ID}"

# Create output folder if needed
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CA5 DATA PROFILING")
print(f"Run ID: {RUN_ID}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("\n[Step 1/7] Loading dataset...")

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
part_cols = ['m_1', 'm_2', 'm_3', 'm_4', 'm_5']

print(f"  Records: {len(df):,}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================
# STEP 2: VALIDATE INVARIANTS
# ============================================================
print("\n[Step 2/7] Validating data rules...")

# Check: exactly 5 parts per day (no missing values)
missing_parts = df[part_cols].isna().sum().sum()
if missing_parts > 0:
    print(f"  WARNING: {missing_parts} missing part values!")
else:
    print("  [OK] All days have 5 parts recorded")

# Check: no duplicates within a day
def check_unique_parts(row):
    parts = row[part_cols].values
    return len(parts) == len(set(parts))

duplicate_mask = ~df.apply(check_unique_parts, axis=1)
num_duplicates = duplicate_mask.sum()
if num_duplicates > 0:
    print(f"  WARNING: {num_duplicates} days have duplicate parts!")
else:
    print("  [OK] All days have 5 UNIQUE parts (no duplicates)")

# Check: all parts in range 1-39
all_parts = df[part_cols].values.flatten()
min_part, max_part = int(all_parts.min()), int(all_parts.max())
if min_part >= 1 and max_part <= 39:
    print(f"  [OK] Part IDs in valid range: [{min_part}, {max_part}]")
else:
    print(f"  WARNING: Parts outside range! Found [{min_part}, {max_part}]")

# ============================================================
# STEP 3: DETECT CALENDAR GAPS
# ============================================================
print("\n[Step 3/7] Detecting calendar gaps...")

df_sorted = df.sort_values('date').reset_index(drop=True)
df_sorted['days_since_last'] = df_sorted['date'].diff().dt.days

gaps = df_sorted[df_sorted['days_since_last'] > 1].copy()
num_gaps = len(gaps)

print(f"  Total gaps found: {num_gaps}")

if num_gaps > 0:
    gap_stats = gaps['days_since_last'].describe()
    print(f"  Gap sizes: min={gap_stats['min']:.0f}, mean={gap_stats['mean']:.1f}, max={gap_stats['max']:.0f} days")

    # Show largest gaps
    print("\n  Largest gaps (top 10):")
    largest_gaps = gaps.nlargest(10, 'days_since_last')[['date', 'days_since_last']]
    for _, row in largest_gaps.iterrows():
        print(f"    {row['date'].date()}: {int(row['days_since_last'])} days gap")
else:
    gap_stats = pd.Series({'min': 0, 'mean': 0, 'max': 0, 'count': 0})

# ============================================================
# STEP 4: ANALYZE PART FREQUENCY
# ============================================================
print("\n[Step 4/7] Analyzing part frequencies...")

part_counts = pd.Series(all_parts).value_counts().sort_index()
expected_freq = len(df) * 5 / 39  # If perfectly uniform

print(f"  Total part usages: {len(all_parts):,}")
print(f"  Unique parts: {len(part_counts)}")
print(f"  Expected frequency (if uniform): {expected_freq:.1f}")
print(f"  Actual range: [{part_counts.min()}, {part_counts.max()}]")
print(f"  Mean: {part_counts.mean():.1f}, Std Dev: {part_counts.std():.1f}")

cv = part_counts.std() / part_counts.mean()
print(f"  Coefficient of variation: {cv:.2%}")
if cv < 0.10:
    print("  [OK] Distribution is near-uniform (CV < 10%)")
else:
    print("  [INFO] Some parts used more than others")

# ============================================================
# STEP 5: TEMPORAL PATTERNS
# ============================================================
print("\n[Step 5/7] Analyzing temporal patterns...")

# Year-by-year record counts
df_sorted['year'] = df_sorted['date'].dt.year
yearly_counts = df_sorted.groupby('year').size()
print(f"  Years covered: {yearly_counts.index.min()} to {yearly_counts.index.max()}")
print(f"  Records per year: min={yearly_counts.min()}, max={yearly_counts.max()}, mean={yearly_counts.mean():.0f}")

# ============================================================
# STEP 6: CONCEPT DRIFT CHECK
# ============================================================
print("\n[Step 6/7] Checking for concept drift...")

try:
    from scipy.stats import ks_2samp

    mid_idx = len(df) // 2
    early_parts = df_sorted.iloc[:mid_idx][part_cols].values.flatten()
    late_parts = df_sorted.iloc[mid_idx:][part_cols].values.flatten()

    ks_stat, p_value = ks_2samp(early_parts, late_parts)
    print(f"  KS Statistic: {ks_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("  [WARNING] Significant distribution shift detected between early and late periods!")
        drift_detected = True
    else:
        print("  [OK] No significant distribution shift")
        drift_detected = False
except ImportError:
    print("  [SKIP] scipy not installed - skipping drift test")
    ks_stat, p_value = None, None
    drift_detected = None

# ============================================================
# STEP 7: GENERATE VISUALIZATIONS AND REPORT
# ============================================================
print("\n[Step 7/7] Generating report and visualizations...")

try:
    import matplotlib.pyplot as plt

    # Plot 1: Part frequency distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(part_counts.index, part_counts.values, color='steelblue', edgecolor='navy')
    ax.axhline(expected_freq, color='red', linestyle='--', linewidth=2, label=f'Expected (uniform): {expected_freq:.0f}')
    ax.set_xlabel('Part ID', fontsize=12)
    ax.set_ylabel('Frequency (times used)', fontsize=12)
    ax.set_title('Part Frequency Distribution (1992-2026)', fontsize=14)
    ax.legend()
    ax.set_xticks(range(1, 40, 2))
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'part_frequency.png', dpi=150)
    plt.close()
    print("  Saved: part_frequency.png")

    # Plot 2: Records per year
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(yearly_counts.index, yearly_counts.values, color='forestgreen', edgecolor='darkgreen')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Records', fontsize=12)
    ax.set_title('Records Per Year', fontsize=14)
    ax.set_xticks(yearly_counts.index[::2])
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'records_per_year.png', dpi=150)
    plt.close()
    print("  Saved: records_per_year.png")

    # Plot 3: Gap distribution
    if num_gaps > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(gaps['days_since_last'], bins=30, color='coral', edgecolor='darkred')
        ax.set_xlabel('Gap Size (days)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Calendar Gaps', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER / 'gap_distribution.png', dpi=150)
        plt.close()
        print("  Saved: gap_distribution.png")

    plots_generated = True
except ImportError:
    print("  [SKIP] matplotlib not installed - skipping plots")
    plots_generated = False

# Save metrics CSV
metrics = {
    'metric': ['record_count', 'date_min', 'date_max', 'total_days_span',
               'gap_count', 'gap_mean', 'gap_max', 'part_freq_mean',
               'part_freq_std', 'part_freq_min', 'part_freq_max', 'cv',
               'ks_statistic', 'ks_pvalue', 'drift_detected'],
    'value': [len(df), str(df['date'].min().date()), str(df['date'].max().date()),
              (df['date'].max() - df['date'].min()).days,
              num_gaps, gap_stats['mean'] if num_gaps > 0 else 0, gap_stats['max'] if num_gaps > 0 else 0,
              part_counts.mean(), part_counts.std(), part_counts.min(), part_counts.max(), cv,
              ks_stat if ks_stat else 'N/A', p_value if p_value else 'N/A',
              str(drift_detected) if drift_detected is not None else 'N/A']
}
pd.DataFrame(metrics).to_csv(OUTPUT_FOLDER / 'metrics.csv', index=False)
print("  Saved: metrics.csv")

# Save config YAML
config_content = f"""# Data Profiling Configuration
run_id: {RUN_ID}
timestamp: {datetime.now().isoformat()}
seed: {SEED}
data_path: {DATA_PATH}
output_folder: {OUTPUT_FOLDER}
"""
with open(OUTPUT_FOLDER / 'config.yaml', 'w') as f:
    f.write(config_content)
print("  Saved: config.yaml")

# Generate report
report = f"""# Data Profiling Report

**Dataset**: CA5_date.csv
**Run ID**: {RUN_ID}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Records | {len(df):,} |
| Date Range | {df['date'].min().date()} to {df['date'].max().date()} |
| Total Days Span | {(df['date'].max() - df['date'].min()).days:,} days |
| Years Covered | {yearly_counts.index.min()} to {yearly_counts.index.max()} |

---

## Data Invariant Validation

| Check | Status |
|-------|--------|
| All days have 5 parts | {'PASS' if missing_parts == 0 else 'FAIL'} |
| No duplicate parts in a day | {'PASS' if num_duplicates == 0 else 'FAIL'} |
| All part IDs in range [1, 39] | {'PASS' if min_part >= 1 and max_part <= 39 else 'FAIL'} |

**Result**: {'All invariants satisfied' if missing_parts == 0 and num_duplicates == 0 else 'Issues detected - review warnings'}

---

## Calendar Gap Analysis

| Metric | Value |
|--------|-------|
| Total Gaps | {num_gaps} |
| Mean Gap Size | {gap_stats['mean']:.1f} days |
| Max Gap Size | {gap_stats['max']:.0f} days |

**Interpretation**: The data has gaps (not every calendar day is recorded). This is expected - likely weekends and holidays are excluded. The largest gaps may indicate extended shutdowns (e.g., COVID period, plant maintenance).

---

## Part Frequency Distribution

| Metric | Value |
|--------|-------|
| Total Part Usages | {len(all_parts):,} |
| Unique Parts | {len(part_counts)} |
| Expected (uniform) | {expected_freq:.1f} |
| Actual Mean | {part_counts.mean():.1f} |
| Actual Std Dev | {part_counts.std():.1f} |
| Coefficient of Variation | {cv:.2%} |

**Interpretation**: {'Distribution is near-uniform - parts are used roughly equally.' if cv < 0.10 else 'Some variation in part usage frequency - certain parts may be more common.'}

---

## Concept Drift Assessment

| Metric | Value |
|--------|-------|
| KS Statistic | {f'{ks_stat:.4f}' if ks_stat else 'N/A'} |
| P-value | {f'{p_value:.4f}' if p_value else 'N/A'} |
| Drift Detected | {'Yes - patterns changed over time' if drift_detected else 'No - patterns stable' if drift_detected is not None else 'N/A'} |

**Interpretation**: {'The distribution of parts has changed significantly between early and late periods. Consider time-aware modeling.' if drift_detected else 'Part usage patterns are relatively stable over time.' if drift_detected is not None else 'Could not assess - scipy not available.'}

---

## Artifacts Generated

- `config.yaml` - Run configuration
- `metrics.csv` - Quantitative metrics
- `part_frequency.png` - Part usage histogram
- `records_per_year.png` - Yearly record counts
- `gap_distribution.png` - Calendar gap sizes
- `data_profile_report.md` - This report

---

## Key Findings for Modeling

1. **Data Quality**: {'Excellent - all invariants satisfied' if missing_parts == 0 and num_duplicates == 0 else 'Review issues above'}
2. **Part Distribution**: {'Near-uniform' if cv < 0.10 else 'Some variation'} - {'' if cv < 0.10 else 'frequency-based baselines may have predictive power'}
3. **Temporal Coverage**: {yearly_counts.index.max() - yearly_counts.index.min() + 1} years of history available
4. **Gaps**: {num_gaps} calendar gaps - feature engineering should account for time gaps
5. **Drift**: {'Consider time-aware models' if drift_detected else 'Standard temporal modeling appropriate' if drift_detected is not None else 'Unknown'}

---

## Recommended Next Steps

1. **Proceed to Baseline Suite** - Build frequency/recency baselines with tier metrics
2. **Use gap information** - Engineer features that handle calendar gaps appropriately
3. **Monitor for drift** - {'Include drift detection in production' if drift_detected else 'Low priority - patterns stable'}

---

**Report generated by Dr. Synapse**
**Workflow**: data-profile
"""

with open(OUTPUT_FOLDER / 'data_profile_report.md', 'w') as f:
    f.write(report)
print("  Saved: data_profile_report.md")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DATA PROFILING COMPLETE")
print("=" * 60)
print(f"\nOutput folder: {OUTPUT_FOLDER}")
print("\nKey Findings:")
print(f"  - Records: {len(df):,}")
print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  - Invariants: {'All passed' if missing_parts == 0 and num_duplicates == 0 else 'Issues found'}")
print(f"  - Calendar gaps: {num_gaps}")
print(f"  - Part distribution: {'Near-uniform' if cv < 0.10 else 'Some variation'} (CV={cv:.1%})")
print(f"  - Concept drift: {'Detected' if drift_detected else 'Not detected' if drift_detected is not None else 'Unknown'}")
print("\nNext step: Run baseline-suite workflow (BL command)")
print("=" * 60)
