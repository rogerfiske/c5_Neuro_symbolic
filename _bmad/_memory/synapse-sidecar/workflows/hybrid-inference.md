# Hybrid Neuro-Symbolic Inference Workflow

**Workflow ID**: hybrid-inference
**Purpose**: Combine neural scores with symbolic rule constraints and stability policies
**Prerequisites**: neural-model-prototype, rulebook-draft completed
**Estimated Duration**: 4-8 hours

---

## Objective

Integrate neural predictions with symbolic rules for interpretable, constrained pool selection:
- Load neural model scores (per-part probabilities)
- Apply symbolic rules to adjust/boost/suppress scores
- Enforce hard constraints (K unique parts, no duplicates)
- Apply stability policy (Jaccard penalty for excessive churn)
- Produce final ranked pool with rule evidence trace

---

## Two-Tier Architecture

### **Tier A: Neural Scoring**
- Neural model produces `p_hat[p]` for each part p (probability of next-day inclusion)
- Output: 39-dimensional score vector

### **Tier B: Symbolic Reasoning + Optimization**
- Apply symbolic rules to adjust scores
- Enforce invariants (5 parts needed, K pool size, uniqueness)
- Optimize for tiered service levels + stability
- Output: Final pool (top-K parts) + rule trace

---

## Symbolic Rule Integration Strategies

### 1. **Score Adjustment**
Modify neural scores based on rule firing:
```python
adjusted_score[p] = neural_score[p] * rule_multiplier[p]
```

Example:
- If cooldown rule fires for part p → multiply score by 0.1 (suppress)
- If co-occurrence rule fires for part q → multiply score by 1.5 (boost)

### 2. **Hard Constraints**
Enforce logical constraints via post-processing:
- Remove parts that violate cooldown rules
- Force inclusion of parts required by high-confidence rules

### 3. **Differentiable Logic** (Advanced)
- Soft constraints via continuous relaxations
- Backpropagate through rule layer during training

---

## Stability Policy

**Problem**: Daily pool changes disrupt operations

**Solution**: Penalize excessive churn
```python
stability_penalty = λ * (1 - Jaccard(pool_today, pool_yesterday))
objective = neural_score + rule_adjustments - stability_penalty
```

**Strong Shift Override**: Allow large changes if confidence is very high

---

## Rule Evidence Trace

For interpretability, log which rules influenced final pool:
```yaml
day: 2024-06-15
pool: [1, 5, 8, 12, 17, 19, 23, 25, 28, 31, 33, 36, 38, ...]
rule_trace:
  - rule_id: cooldown_part_5
    action: suppressed part 5
    confidence: 0.92
  - rule_id: cooccur_10_17
    action: boosted part 17 (given part 10 in recent history)
    confidence: 0.68
  - stability_policy: maintained 18/23 parts from yesterday (Jaccard=0.78)
```

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/hybrid-inference/{run-id}/`

**Files**:
- `config.yaml` - Rule weights, stability λ, integration strategy
- `metrics.csv` - Tier rates, stability metrics, rule firing frequencies
- `pool_predictions.csv` - Daily pools with neural scores, adjusted scores, rule trace
- `rule_impact_analysis.png` - Visualization of rule contributions
- `hybrid_vs_neural.png` - Comparison with pure neural baseline

---

## Success Criteria

✅ Neural + symbolic integration implemented
✅ Rule adjustments improve tier metrics OR interpretability
✅ Stability policy reduces churn (higher Jaccard vs pure neural)
✅ Rule evidence traces generated for all predictions
✅ Beats pure neural model on Good-or-better rate (or matches with better stability)

---

## Next Workflow

**k-optimizer** - Choose optimal K under tiered constraints

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
