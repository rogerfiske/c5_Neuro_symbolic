"""
Rulebook Draft - Symbolic Rule Discovery for CA5 Dataset
=========================================================
Run ID: run-001
Purpose: Discover interpretable rules that can guide predictions

What this script does (in plain English):
1. Looks for COOLDOWN patterns (how long before a part is reused)
2. Looks for CO-OCCURRENCE patterns (parts that appear together)
3. Looks for BURST patterns (parts that appear multiple days in a row)
4. Validates each rule with statistics (how reliable is it?)
5. Creates a rulebook that can be used with neural models

To run: python scripts/rulebook_draft.py
Estimated time: 5-10 minutes
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

NUM_PARTS = 39

# Rule mining parameters (relaxed to find weaker patterns)
MIN_SUPPORT = 30  # Minimum instances for a rule to be valid
MIN_CONFIDENCE = 0.15  # Minimum confidence threshold
MIN_LIFT = 1.1  # Minimum lift (surprise factor)

# Paths
PROJECT_ROOT = Path("C:/Users/Minis/CascadeProjects/c5_neuro_symbolic")
DATA_PATH = PROJECT_ROOT / "data/raw/CA5_date.csv"
OUTPUT_FOLDER = PROJECT_ROOT / f"_bmad-output/synapse/rulebook-draft/{RUN_ID}"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("RULEBOOK DRAFT - SYMBOLIC RULE DISCOVERY")
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

print(f"  Records: {len(df):,}")

# Build part sets per day
part_sets = []
for _, row in df.iterrows():
    parts = set(int(row[col]) for col in part_cols)
    part_sets.append(parts)

print(f"  Part sets built: {len(part_sets)}")

# ============================================================
# RULE TYPE 1: COOLDOWN RULES
# ============================================================
print("\n[2/6] Mining cooldown rules...")
print("  (How long before a part is reused after being used)")

# Track time between consecutive uses for each part
reuse_gaps = defaultdict(list)
last_used_idx = {p: -9999 for p in range(1, NUM_PARTS + 1)}

for idx, parts in enumerate(part_sets):
    for p in parts:
        if last_used_idx[p] >= 0:
            gap = idx - last_used_idx[p]
            reuse_gaps[p].append(gap)
        last_used_idx[p] = idx

# Analyze cooldown patterns
cooldown_rules = []
for part_id in range(1, NUM_PARTS + 1):
    gaps = reuse_gaps[part_id]
    if len(gaps) >= MIN_SUPPORT:
        gaps_arr = np.array(gaps)
        median_gap = np.median(gaps_arr)
        mean_gap = np.mean(gaps_arr)

        # Cooldown = period where P(reuse) is very low
        # Count how many reuses happen within different windows
        within_3 = (gaps_arr <= 3).sum() / len(gaps_arr)
        within_5 = (gaps_arr <= 5).sum() / len(gaps_arr)
        within_7 = (gaps_arr <= 7).sum() / len(gaps_arr)

        # If very few reuses within 3 days, there's a cooldown effect
        if within_3 < 0.25:  # Less than 25% reuse within 3 days
            cooldown_days = 3
            confidence = 1 - within_3
        elif within_5 < 0.35:
            cooldown_days = 5
            confidence = 1 - within_5
        elif within_7 < 0.45:
            cooldown_days = 7
            confidence = 1 - within_7
        else:
            cooldown_days = None
            confidence = 0

        if cooldown_days and confidence >= MIN_CONFIDENCE:
            cooldown_rules.append({
                'part_id': part_id,
                'cooldown_days': cooldown_days,
                'confidence': confidence,
                'support': len(gaps),
                'median_gap': median_gap,
                'mean_gap': mean_gap
            })

print(f"  Cooldown rules found: {len(cooldown_rules)}")

# ============================================================
# RULE TYPE 2: CO-OCCURRENCE RULES (Same Day)
# ============================================================
print("\n[3/6] Mining co-occurrence rules (same day)...")
print("  (Parts that tend to appear together)")

# Count co-occurrences
cooccur_counts = defaultdict(int)
part_counts = defaultdict(int)

for parts in part_sets:
    for p in parts:
        part_counts[p] += 1
    # Count pairs
    parts_list = sorted(parts)
    for i, p1 in enumerate(parts_list):
        for p2 in parts_list[i+1:]:
            cooccur_counts[(p1, p2)] += 1

n_days = len(part_sets)

# Calculate co-occurrence rules
cooccur_rules = []
for (p1, p2), count in cooccur_counts.items():
    if count >= MIN_SUPPORT:
        # P(p2 | p1) = P(p1 and p2) / P(p1)
        prob_p1 = part_counts[p1] / n_days
        prob_p2 = part_counts[p2] / n_days
        prob_both = count / n_days

        confidence_p1_to_p2 = prob_both / prob_p1 if prob_p1 > 0 else 0
        confidence_p2_to_p1 = prob_both / prob_p2 if prob_p2 > 0 else 0

        # Lift = P(p2|p1) / P(p2) = how much more likely p2 is when p1 is present
        lift = confidence_p1_to_p2 / prob_p2 if prob_p2 > 0 else 0

        if lift >= MIN_LIFT and max(confidence_p1_to_p2, confidence_p2_to_p1) >= MIN_CONFIDENCE:
            cooccur_rules.append({
                'part_1': p1,
                'part_2': p2,
                'count': count,
                'confidence_1_to_2': confidence_p1_to_p2,
                'confidence_2_to_1': confidence_p2_to_p1,
                'lift': lift,
                'support': count
            })

# Sort by lift
cooccur_rules.sort(key=lambda x: -x['lift'])
print(f"  Co-occurrence rules found: {len(cooccur_rules)}")

# ============================================================
# RULE TYPE 3: SEQUENTIAL RULES (Next Day)
# ============================================================
print("\n[4/6] Mining sequential rules (day-to-day)...")
print("  (If Part A today, what's likely tomorrow?)")

# Count sequential patterns
seq_counts = defaultdict(int)  # (p_today, p_tomorrow) -> count
today_counts = defaultdict(int)

for idx in range(len(part_sets) - 1):
    today_parts = part_sets[idx]
    tomorrow_parts = part_sets[idx + 1]

    for p_today in today_parts:
        today_counts[p_today] += 1
        for p_tomorrow in tomorrow_parts:
            seq_counts[(p_today, p_tomorrow)] += 1

# Calculate sequential rules
sequential_rules = []
for (p_today, p_tomorrow), count in seq_counts.items():
    if count >= MIN_SUPPORT:
        # P(p_tomorrow | p_today)
        confidence = count / today_counts[p_today] if today_counts[p_today] > 0 else 0

        # Expected probability of p_tomorrow (baseline)
        p_tomorrow_baseline = part_counts[p_tomorrow] / n_days

        # Lift
        lift = confidence / p_tomorrow_baseline if p_tomorrow_baseline > 0 else 0

        if lift >= MIN_LIFT and confidence >= MIN_CONFIDENCE:
            sequential_rules.append({
                'part_today': p_today,
                'part_tomorrow': p_tomorrow,
                'count': count,
                'confidence': confidence,
                'baseline_prob': p_tomorrow_baseline,
                'lift': lift,
                'support': count
            })

sequential_rules.sort(key=lambda x: -x['lift'])
print(f"  Sequential rules found: {len(sequential_rules)}")

# ============================================================
# RULE TYPE 4: BURST RULES (Consecutive Days)
# ============================================================
print("\n[5/6] Mining burst rules...")
print("  (Parts that appear multiple days in a row)")

# Track consecutive appearances
burst_stats = defaultdict(list)  # part -> list of burst lengths

for p in range(1, NUM_PARTS + 1):
    current_streak = 0
    for idx, parts in enumerate(part_sets):
        if p in parts:
            current_streak += 1
        else:
            if current_streak > 0:
                burst_stats[p].append(current_streak)
            current_streak = 0
    if current_streak > 0:
        burst_stats[p].append(current_streak)

# Analyze burst patterns
burst_rules = []
for part_id in range(1, NUM_PARTS + 1):
    bursts = burst_stats[part_id]
    if len(bursts) >= MIN_SUPPORT:
        bursts_arr = np.array(bursts)

        # Calculate probability of continuation
        # If part appears today, what's P(appears tomorrow)?
        total_appearances = part_counts[part_id]
        multi_day_bursts = (bursts_arr >= 2).sum()

        # Count transitions: part today -> part tomorrow
        cont_count = 0
        for idx in range(len(part_sets) - 1):
            if part_id in part_sets[idx] and part_id in part_sets[idx + 1]:
                cont_count += 1

        # Continuation probability
        appearances_with_tomorrow = sum(1 for idx in range(len(part_sets) - 1) if part_id in part_sets[idx])
        cont_prob = cont_count / appearances_with_tomorrow if appearances_with_tomorrow > 0 else 0

        # Baseline: overall probability of part appearing
        baseline_prob = part_counts[part_id] / n_days

        # Lift for burst
        lift = cont_prob / baseline_prob if baseline_prob > 0 else 0

        if cont_prob >= 0.15 and lift >= MIN_LIFT:  # 15%+ chance of consecutive days
            burst_rules.append({
                'part_id': part_id,
                'continuation_prob': cont_prob,
                'baseline_prob': baseline_prob,
                'lift': lift,
                'avg_burst_length': np.mean(bursts_arr),
                'max_burst_length': np.max(bursts_arr),
                'num_bursts': len(bursts),
                'support': total_appearances
            })

burst_rules.sort(key=lambda x: -x['lift'])
print(f"  Burst rules found: {len(burst_rules)}")

# ============================================================
# COMPILE RULEBOOK
# ============================================================
print("\n[6/6] Compiling rulebook...")

all_rules = []
rule_id = 1

# Add cooldown rules
for rule in cooldown_rules[:10]:  # Top 10
    all_rules.append({
        'id': f'COOL_{rule_id:03d}',
        'type': 'cooldown',
        'part_id': rule['part_id'],
        'description': f"Part {rule['part_id']} has {rule['cooldown_days']}-day cooldown",
        'formula': f"P(Part {rule['part_id']} within {rule['cooldown_days']} days of last use) < {1-rule['confidence']:.0%}",
        'confidence': rule['confidence'],
        'support': rule['support'],
        'lift': None,
        'details': rule
    })
    rule_id += 1

# Add co-occurrence rules
for rule in cooccur_rules[:10]:  # Top 10
    all_rules.append({
        'id': f'COOC_{rule_id:03d}',
        'type': 'co-occurrence',
        'part_id': (rule['part_1'], rule['part_2']),
        'description': f"Parts {rule['part_1']} and {rule['part_2']} tend to co-occur",
        'formula': f"P(Part {rule['part_2']} | Part {rule['part_1']}) = {rule['confidence_1_to_2']:.0%}",
        'confidence': max(rule['confidence_1_to_2'], rule['confidence_2_to_1']),
        'support': rule['support'],
        'lift': rule['lift'],
        'details': rule
    })
    rule_id += 1

# Add sequential rules
for rule in sequential_rules[:10]:  # Top 10
    all_rules.append({
        'id': f'SEQ_{rule_id:03d}',
        'type': 'sequential',
        'part_id': (rule['part_today'], rule['part_tomorrow']),
        'description': f"Part {rule['part_today']} today -> Part {rule['part_tomorrow']} tomorrow",
        'formula': f"P(Part {rule['part_tomorrow']} tomorrow | Part {rule['part_today']} today) = {rule['confidence']:.0%}",
        'confidence': rule['confidence'],
        'support': rule['support'],
        'lift': rule['lift'],
        'details': rule
    })
    rule_id += 1

# Add burst rules
for rule in burst_rules[:10]:  # Top 10
    all_rules.append({
        'id': f'BURST_{rule_id:03d}',
        'type': 'burst',
        'part_id': rule['part_id'],
        'description': f"Part {rule['part_id']} tends to appear consecutive days",
        'formula': f"P(Part {rule['part_id']} tomorrow | Part {rule['part_id']} today) = {rule['continuation_prob']:.0%}",
        'confidence': rule['continuation_prob'],
        'support': rule['support'],
        'lift': rule['lift'],
        'details': rule
    })
    rule_id += 1

print(f"  Total rules in rulebook: {len(all_rules)}")

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\n  Saving outputs...")

# Save rule evidence CSV
evidence_rows = []
for rule in all_rules:
    evidence_rows.append({
        'rule_id': rule['id'],
        'type': rule['type'],
        'description': rule['description'],
        'confidence': rule['confidence'],
        'support': rule['support'],
        'lift': rule['lift']
    })

evidence_df = pd.DataFrame(evidence_rows)
evidence_df.to_csv(OUTPUT_FOLDER / 'rule_evidence.csv', index=False)
print(f"  Saved: rule_evidence.csv")

# Save rulebook YAML
rulebook_yaml = f"""# Symbolic Rulebook - CA5 Predictive Maintenance
# Generated: {datetime.now().isoformat()}
# Run ID: {RUN_ID}

metadata:
  total_rules: {len(all_rules)}
  rule_types:
    cooldown: {len([r for r in all_rules if r['type'] == 'cooldown'])}
    co_occurrence: {len([r for r in all_rules if r['type'] == 'co-occurrence'])}
    sequential: {len([r for r in all_rules if r['type'] == 'sequential'])}
    burst: {len([r for r in all_rules if r['type'] == 'burst'])}
  min_support: {MIN_SUPPORT}
  min_confidence: {MIN_CONFIDENCE}
  min_lift: {MIN_LIFT}

rules:
"""

for rule in all_rules:
    rulebook_yaml += f"""
  - id: {rule['id']}
    type: {rule['type']}
    description: "{rule['description']}"
    formula: "{rule['formula']}"
    confidence: {rule['confidence']:.3f}
    support: {rule['support']}
    lift: {rule['lift'] if rule['lift'] else 'null'}
"""

with open(OUTPUT_FOLDER / 'rulebook.yaml', 'w', encoding='utf-8') as f:
    f.write(rulebook_yaml)
print(f"  Saved: rulebook.yaml")

# Generate visualizations
try:
    import matplotlib.pyplot as plt

    # Plot 1: Rule confidence distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cooldown rules
    if cooldown_rules:
        ax = axes[0, 0]
        parts = [r['part_id'] for r in cooldown_rules[:15]]
        confs = [r['confidence'] for r in cooldown_rules[:15]]
        ax.barh(range(len(parts)), confs, color='steelblue')
        ax.set_yticks(range(len(parts)))
        ax.set_yticklabels([f'Part {p}' for p in parts])
        ax.set_xlabel('Confidence')
        ax.set_title('Cooldown Rules (Top 15)')
        ax.set_xlim(0, 1)

    # Co-occurrence rules
    if cooccur_rules:
        ax = axes[0, 1]
        pairs = [f"{r['part_1']}-{r['part_2']}" for r in cooccur_rules[:15]]
        lifts = [r['lift'] for r in cooccur_rules[:15]]
        ax.barh(range(len(pairs)), lifts, color='forestgreen')
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs)
        ax.set_xlabel('Lift')
        ax.set_title('Co-occurrence Rules (Top 15 by Lift)')

    # Sequential rules
    if sequential_rules:
        ax = axes[1, 0]
        pairs = [f"{r['part_today']}->{r['part_tomorrow']}" for r in sequential_rules[:15]]
        lifts = [r['lift'] for r in sequential_rules[:15]]
        ax.barh(range(len(pairs)), lifts, color='coral')
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs)
        ax.set_xlabel('Lift')
        ax.set_title('Sequential Rules (Top 15 by Lift)')

    # Burst rules
    if burst_rules:
        ax = axes[1, 1]
        parts = [r['part_id'] for r in burst_rules[:15]]
        probs = [r['continuation_prob'] for r in burst_rules[:15]]
        ax.barh(range(len(parts)), probs, color='purple')
        ax.set_yticks(range(len(parts)))
        ax.set_yticklabels([f'Part {p}' for p in parts])
        ax.set_xlabel('Continuation Probability')
        ax.set_title('Burst Rules (Top 15)')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / 'rule_summary.png', dpi=150)
    plt.close()
    print(f"  Saved: rule_summary.png")

except ImportError:
    print("  [SKIP] matplotlib not installed")

# Generate report
report = f"""# Rulebook Draft Report

**Run ID**: {RUN_ID}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: CA5_date.csv ({len(df):,} days)

---

## Executive Summary

Discovered **{len(all_rules)} interpretable rules** across 4 categories:

| Rule Type | Count | Description |
|-----------|-------|-------------|
| Cooldown | {len([r for r in all_rules if r['type'] == 'cooldown'])} | Parts have reuse delays |
| Co-occurrence | {len([r for r in all_rules if r['type'] == 'co-occurrence'])} | Parts appear together |
| Sequential | {len([r for r in all_rules if r['type'] == 'sequential'])} | Part A today -> Part B tomorrow |
| Burst | {len([r for r in all_rules if r['type'] == 'burst'])} | Parts appear consecutive days |

---

## Top Rules by Type

### Cooldown Rules (Parts have reuse delays)

These parts tend NOT to be reused quickly after being used:

| Part | Cooldown | Confidence | Support |
|------|----------|------------|---------|
"""

for rule in [r for r in all_rules if r['type'] == 'cooldown'][:5]:
    report += f"| Part {rule['details']['part_id']} | {rule['details']['cooldown_days']} days | {rule['confidence']:.0%} | {rule['support']} |\n"

report += """
### Co-occurrence Rules (Parts appear together)

These part pairs tend to appear on the same day:

| Part Pair | Lift | Confidence | Support |
|-----------|------|------------|---------|
"""

for rule in [r for r in all_rules if r['type'] == 'co-occurrence'][:5]:
    d = rule['details']
    report += f"| {d['part_1']} & {d['part_2']} | {d['lift']:.1f}x | {rule['confidence']:.0%} | {rule['support']} |\n"

report += """
### Sequential Rules (Day-to-day patterns)

If Part A appears today, Part B is more likely tomorrow:

| Pattern | Lift | Confidence | Support |
|---------|------|------------|---------|
"""

for rule in [r for r in all_rules if r['type'] == 'sequential'][:5]:
    d = rule['details']
    report += f"| {d['part_today']} -> {d['part_tomorrow']} | {d['lift']:.1f}x | {rule['confidence']:.0%} | {rule['support']} |\n"

report += """
### Burst Rules (Consecutive day patterns)

These parts tend to appear multiple days in a row:

| Part | Continuation Prob | Lift | Avg Burst |
|------|-------------------|------|-----------|
"""

for rule in [r for r in all_rules if r['type'] == 'burst'][:5]:
    d = rule['details']
    report += f"| Part {d['part_id']} | {d['continuation_prob']:.0%} | {d['lift']:.1f}x | {d['avg_burst_length']:.1f} days |\n"

report += f"""
---

## How to Use These Rules

### In Hybrid Inference:

1. **Cooldown rules**: Penalize parts that were used recently
   - If Part X was used 2 days ago and has 5-day cooldown, reduce its score

2. **Co-occurrence rules**: Boost parts associated with today's selections
   - If we've already selected Part 10, boost Part 17 (if they co-occur)

3. **Sequential rules**: Boost parts that follow yesterday's parts
   - If Part A was used yesterday, boost Part B

4. **Burst rules**: Boost parts from recent consecutive runs
   - If Part X appeared yesterday and has burst tendency, boost it

### Constraint Satisfaction:

- Rules can be encoded as **soft constraints** (preferences) or **hard constraints** (must satisfy)
- Combine with neural scores: `final_score = neural_score + rule_bonus - rule_penalty`

---

## Rule Quality Assessment

| Metric | Value |
|--------|-------|
| Min Support Threshold | {MIN_SUPPORT} |
| Min Confidence Threshold | {MIN_CONFIDENCE} |
| Min Lift Threshold | {MIN_LIFT} |
| Rules Meeting All Thresholds | {len(all_rules)} |

---

## Next Steps

1. **Neural Model Prototype** (NP) - Build neural scorer to rank parts
2. **Hybrid Inference** (HI) - Combine neural scores with these rules
3. **Evaluate** - Test if rules improve on 53.1% baseline

---

**Report generated by Dr. Synapse**
**Workflow**: rulebook-draft
"""

with open(OUTPUT_FOLDER / 'rulebook_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  Saved: rulebook_report.md")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("RULEBOOK DRAFT COMPLETE")
print("=" * 70)
print(f"\nOutput folder: {OUTPUT_FOLDER}")
print(f"\n{'='*50}")
print("RULES DISCOVERED")
print(f"{'='*50}")
print(f"""
  Cooldown rules:     {len([r for r in all_rules if r['type'] == 'cooldown'])} (parts with reuse delays)
  Co-occurrence:      {len([r for r in all_rules if r['type'] == 'co-occurrence'])} (parts that appear together)
  Sequential:         {len([r for r in all_rules if r['type'] == 'sequential'])} (day-to-day patterns)
  Burst rules:        {len([r for r in all_rules if r['type'] == 'burst'])} (consecutive day patterns)
  -----------------------------------------
  TOTAL:              {len(all_rules)} interpretable rules

  These rules will be combined with neural models in hybrid inference.
""")
print("=" * 70)
