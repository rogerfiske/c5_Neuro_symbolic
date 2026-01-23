# Feature Schema Engineering Workflow

**Workflow ID**: feature-schema
**Purpose**: Define features + transformations with rigorous leakage audits
**Prerequisites**: data-profile, baseline-suite completed
**Estimated Duration**: 3-6 hours

---

## Objective

Design feature representations for neural/symbolic models:
- Multi-hot vector encoding (39-dim binary vector per day)
- Recency features (time since last use per part)
- Temporal features (day-of-week, month, holiday indicators)
- Gap features (explicit representation of skipped calendar days)
- Association features (part co-occurrence statistics)
- Sequence motifs (n-gram patterns in daily part sets)
- **CRITICAL**: Audit ALL features for future leakage

---

## Feature Categories

### 1. **Core Representations**
- Multi-hot vector: `x_t[p]=1` if part p used on day t, else 0
- One-hot part ID: For per-part models

### 2. **Recency Features**
- Time since last use (TSLU) per part
- Days until next use (DTNU) - **LEAKAGE RISK**: only for post-hoc analysis
- Burst indicators: consecutive days used

### 3. **Temporal Context**
- Day of week (Mon=0, ..., Sun=6)
- Month (1-12)
- Days since epoch (linear time trend)
- Calendar gap indicator (days since last record)

### 4. **Association Features**
- Co-occurrence counts (parts used together)
- Conditional probabilities P(q|p) over sliding windows
- Part centrality in association graph

### 5. **Sequence Motifs**
- 2-gram, 3-gram of part sets
- Motif frequency in recent history

---

## Leakage Audit Protocol

**FOR EVERY FEATURE**:
1. Can this feature be computed from past data only? (YES = OK, NO = LEAKAGE)
2. Does feature computation window extend into future? (NO = OK, YES = LEAKAGE)
3. Is feature value influenced by target day's true parts? (NO = OK, YES = LEAKAGE)

**Red Flags**:
- Using target day's data in any way
- Forward-looking windows (e.g., "next 7 days")
- Global statistics not time-windowed

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/feature-schema/{run-id}/`

**Files**:
- `feature_definitions.yaml` - Complete feature catalog with leakage audit status
- `feature_examples.csv` - Sample feature vectors for validation
- `leakage_audit_report.md` - Detailed leakage analysis
- `feature_correlation.png` - Feature correlation heatmap

---

## Success Criteria

✅ Multi-hot encoding validated
✅ All recency features defined and tested
✅ Temporal features added
✅ Association features computed
✅ **Leakage audit passed for ALL features**
✅ Feature catalog documented

---

## Next Workflow

**rulebook-draft** - Discover symbolic rules with evidence

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
