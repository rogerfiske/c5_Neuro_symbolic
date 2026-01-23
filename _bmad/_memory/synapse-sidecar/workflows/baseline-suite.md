# Baseline Suite Development Workflow

**Workflow ID**: baseline-suite
**Purpose**: Build strong baseline models with tier metrics across K sweep
**Prerequisites**: data-profile workflow completed
**Estimated Duration**: 2-4 hours

---

## Objective

Establish performance floor before neural/symbolic models:
- Implement frequency-based ranking (global part counts)
- Implement recency-based ranking (time since last use)
- Implement co-occurrence heuristics (part association patterns)
- Evaluate all baselines with tier metrics (Excellent/Good/Unacceptable) for K ∈ [20, 27]
- Identify best baseline as benchmark to beat

---

## Baseline Models to Implement

### 1. **Frequency Baseline**
Rank parts by global frequency, predict top-K most common parts

### 2. **Recency Baseline**
Rank parts by time since last use (shorter = higher priority)

### 3. **Last-N-Days Baseline**
Rank by frequency in last N days (N=7, 14, 30)

### 4. **Co-Occurrence Baseline**
Build part association graph, rank by conditional probability given recent parts

### 5. **Weighted Recency-Frequency**
Combine recency + frequency with learned weights

---

## Evaluation Protocol

**Rolling-origin backtesting**:
- Train on data up to day T
- Predict pool for day T+1
- Evaluate tier metrics
- Slide window forward, repeat

**Metrics**:
- Excellent rate @K (5/5 in pool)
- Good-or-better rate @K (≥4/5 in pool)
- Unacceptable rate @K (≤3/5 in pool)
- Pool stability (Jaccard similarity day-to-day)

**K Sweep**: Test K ∈ [20, 21, 22, ..., 27]

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/baseline-suite/{run-id}/`

**Files**:
- `config.yaml` - Baseline configurations
- `metrics.csv` - Tier rates for all baselines × all K values
- `baseline_comparison.png` - Bar charts of Good-or-better rates
- `k_sweep_analysis.png` - Performance vs K for each baseline
- `baseline_report.md` - Summary and recommendations

---

## Success Criteria

✅ All 5 baselines implemented and tested
✅ Tier metrics computed for K ∈ [20, 27]
✅ Best baseline identified (highest Good-or-better rate)
✅ Stability metrics computed
✅ Comprehensive comparison report generated

---

## Next Workflow

**feature-schema** - Engineer features with leakage audits

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
