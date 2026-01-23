# Rulebook Draft Workflow

**Workflow ID**: rulebook-draft
**Purpose**: Propose symbolic rules + tests + evidence plan
**Prerequisites**: feature-schema completed
**Estimated Duration**: 4-8 hours

---

## Objective

Discover interpretable symbolic rules for part selection:
- Co-occurrence rules ("if p, then likely q within Δ days")
- Cooldown rules ("part p unlikely within N days of last use")
- Burst rules ("part p tends to appear M consecutive days")
- Mutual exclusion rules ("p and q rarely co-occur")
- Regime-based rules ("in state S, prioritize subset X")
- Validate rules with confidence/support metrics

---

## Rule Mining Approaches

### 1. **Association Rule Mining**
- Use Apriori or FP-Growth for frequent itemsets
- Extract rules: {p1, p2} → {p3} with confidence threshold
- Focus on temporal associations (within-day, cross-day)

### 2. **Temporal Pattern Mining**
- Identify sequential patterns: p → q (within Δ days)
- Cooldown period estimation: histogram of TSLU when part recurs

### 3. **Constraint Discovery**
- Mutual exclusion: P(p ∧ q same day) ≈ 0?
- Complementary sets: parts that tend to "fill out" to 5

### 4. **Regime Detection** (optional)
- Cluster days by part usage patterns
- Learn regime-specific rules

---

## Rule Validation

**For each candidate rule**:
1. **Support**: How often does rule antecedent appear?
2. **Confidence**: P(consequent | antecedent)?
3. **Lift**: Confidence / P(consequent) - measures surprise
4. **Coverage**: % of days rule applies to
5. **Exceptions**: Days where rule fires but fails

---

## Example Rules

```yaml
rules:
  - id: cooldown_part_5
    type: cooldown
    description: "Part 5 has 7-day cooldown period"
    formula: "if part 5 used on day t, P(part 5 on day t+k) ≈ 0 for k < 7"
    confidence: 0.92
    support: 1847 instances

  - id: cooccur_10_17
    type: co-occurrence
    description: "Part 10 and Part 17 often co-occur"
    formula: "P(17 | 10) = 0.68, P(17) = 0.13"
    lift: 5.2
    confidence: 0.68

  - id: burst_part_23
    type: burst
    description: "Part 23 tends to appear 3 consecutive days"
    formula: "if part 23 on day t, P(part 23 on t+1) = 0.71"
    confidence: 0.71
```

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/rulebook-draft/{run-id}/`

**Files**:
- `rulebook.yaml` - Complete rule catalog with validation metrics
- `rule_evidence.csv` - Support/confidence/lift for all rules
- `rule_examples.md` - Concrete examples of rule firing
- `rule_validation_report.md` - Comprehensive analysis

---

## Success Criteria

✅ ≥10 candidate rules discovered
✅ All rules validated with support/confidence/lift
✅ Rule conflicts identified (if any)
✅ Evidence plan for neuro-symbolic integration defined
✅ Rulebook documented in structured format

---

## Next Workflow

**neural-model-prototype** - Build calibrated neural scorers

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
