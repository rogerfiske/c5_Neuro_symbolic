# Pool Size (K) Optimization Workflow

**Workflow ID**: k-optimizer
**Purpose**: Choose optimal K under tiered service constraints and stability penalty
**Prerequisites**: hybrid-inference completed
**Estimated Duration**: 2-4 hours

---

## Objective

Determine optimal pool size K* ∈ [20, 27] that:
- Maximizes Good-or-better rate (≥4/5 true parts in pool)
- Minimizes Unacceptable rate (≤3/5 true parts in pool)
- Minimizes pool size K (inventory cost proxy)
- Maintains reasonable stability (Jaccard ≥ threshold)

---

## Optimization Approaches

### 1. **Constrained Optimization**
Choose smallest K satisfying constraints:
```
Minimize: K
Subject to:
  P(Good-or-better | K) ≥ S_target  (e.g., 90%)
  P(Unacceptable | K) ≤ U_max       (e.g., 5%)
  Jaccard(pool_t, pool_{t-1}) ≥ J_min (e.g., 0.70)
```

### 2. **Expected Cost Minimization**
Define cost function:
```
Cost(K) = C_stock(K) + E[C_tier(m) | K] + C_thrash(K)

Where:
- C_stock(K) = α * K  (inventory cost)
- C_tier(m) = cost based on # missing parts m
  - m=0 (5/5): 0
  - m=1 (4/5): β  (expedited single part)
  - m≥2 (≤3/5): γ (very large penalty)
- C_thrash(K) = δ * (1 - Jaccard)
```

Choose K* = argmin_{K} Cost(K)

### 3. **Pareto Frontier Analysis**
Plot tradeoff curves:
- K vs Good-or-better rate
- K vs Unacceptable rate
- K vs Jaccard stability

Identify Pareto-optimal K values

---

## Evaluation Protocol

**For each K ∈ [20, 21, 22, ..., 27]**:
1. Run hybrid-inference with pool size K
2. Compute tier rates on validation set (rolling-origin)
3. Compute stability metrics (Jaccard day-to-day)
4. Estimate expected cost or constraint satisfaction

**Statistical Validation**: Bootstrap confidence intervals for tier rates

---

## Tier-Based Cost Function (Example)

```yaml
cost_structure:
  inventory_cost_per_part: 10  # α
  excellent_penalty: 0          # m=0
  good_penalty: 50              # m=1 (expedite cost)
  acceptable_penalty: 200       # m=2 (expedite 2 parts)
  unacceptable_penalty: 1000    # m≥3 (production disruption)
  stability_penalty_per_churn: 5  # δ per part changed
```

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/k-optimizer/{run-id}/`

**Files**:
- `config.yaml` - Cost structure, constraints, K range
- `metrics.csv` - Tier rates and costs for all K values
- `k_optimization_report.md` - Analysis and recommendation
- `pareto_frontier.png` - K vs tier rates tradeoff curve
- `cost_curve.png` - Expected cost vs K
- `optimal_k_summary.yaml` - Final recommendation with justification

---

## Success Criteria

✅ All K ∈ [20, 27] evaluated with tier metrics
✅ Optimal K* identified based on constraints OR cost minimization
✅ Statistical confidence intervals computed
✅ Sensitivity analysis performed (vary cost parameters)
✅ Clear recommendation documented with evidence

---

## Next Workflow

**ablation-report** - Systematic comparison of all variants

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
