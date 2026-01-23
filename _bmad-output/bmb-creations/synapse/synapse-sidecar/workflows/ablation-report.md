# Ablation & Comparative Analysis Workflow

**Workflow ID**: ablation-report
**Purpose**: Systematically compare all variants and produce evidence-based conclusions
**Prerequisites**: All prior workflows completed
**Estimated Duration**: 3-6 hours

---

## Objective

Conduct rigorous ablation study to understand contribution of each component:
- Compare all baselines, neural models, hybrid variants
- Ablate components (remove rules, remove neural, remove stability)
- Statistical significance testing (bootstrap, permutation tests)
- Produce comprehensive research conclusion with evidence
- Recommend production-viable approach OR identify research gaps

---

## Ablation Matrix

| Variant | Neural | Rules | Stability | K* | Good-or-better | Unacceptable | Jaccard |
|---------|--------|-------|-----------|-----|----------------|--------------|---------|
| Frequency Baseline | ❌ | ❌ | ❌ | - | TBD | TBD | TBD |
| Recency Baseline | ❌ | ❌ | ❌ | - | TBD | TBD | TBD |
| Co-occurrence Baseline | ❌ | ❌ | ❌ | - | TBD | TBD | TBD |
| Logistic Regression | ✅ | ❌ | ❌ | - | TBD | TBD | TBD |
| GRU | ✅ | ❌ | ❌ | - | TBD | TBD | TBD |
| Transformer | ✅ | ❌ | ❌ | - | TBD | TBD | TBD |
| Hybrid (Neural + Rules) | ✅ | ✅ | ❌ | - | TBD | TBD | TBD |
| Hybrid + Stability | ✅ | ✅ | ✅ | K* | TBD | TBD | TBD |
| Hybrid w/o Rules | ✅ | ❌ | ✅ | - | TBD | TBD | TBD |
| Hybrid w/o Stability | ✅ | ✅ | ❌ | - | TBD | TBD | TBD |

---

## Statistical Testing

**Pairwise Comparisons**:
- Bootstrap confidence intervals (95%) for tier rates
- McNemar's test for paired predictions (day-level)
- Permutation test for distribution differences

**Significance Threshold**: p < 0.05 (with Bonferroni correction for multiple comparisons)

---

## Key Questions to Answer

1. **Do neural models beat baselines?**
   - If NO → Why? Feature engineering issue? Dataset too uniform?
   - If YES → By how much? Is improvement practically significant?

2. **Do symbolic rules add value?**
   - Compare Hybrid vs Pure Neural
   - Are rule adjustments helping or hurting?
   - Do rules improve interpretability even if metrics similar?

3. **Does stability policy work?**
   - Does it reduce churn (higher Jaccard)?
   - Does it maintain or improve tier rates?
   - What is optimal λ (stability weight)?

4. **What is optimal K*?**
   - Is K* consistent across variants?
   - Trade-off: smaller K (lower cost) vs higher coverage?

5. **Production readiness?**
   - Can we deploy this approach?
   - What are remaining research gaps?
   - What would move this to production?

---

## Error Analysis

**Near-Miss Analysis**: Days where pool was 4/5 (Good) but not 5/5 (Excellent)
- Which part was missed?
- Why did model rank it low?
- Were there rule conflicts?

**Failure Analysis**: Days where pool was ≤3/5 (Unacceptable)
- What went wrong?
- Outlier days? Concept drift?
- Model overconfidence?

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/ablation-report/{run-id}/`

**Files**:
- `ablation_matrix.csv` - All variants with complete metrics
- `statistical_tests.csv` - P-values, confidence intervals for all comparisons
- `ablation_report.md` - Comprehensive analysis answering all key questions
- `comparison_charts.png` - Bar charts, box plots for visual comparison
- `error_analysis.md` - Deep dive into failures and near-misses
- `production_recommendation.md` - Final verdict: deploy, iterate, or research further

---

## Research Conclusion Template

```markdown
# Neuro-Symbolic Predictive Maintenance: Research Conclusion

## Executive Summary
[1-2 paragraphs: What did we build? What did we learn? What's the recommendation?]

## Key Findings
1. **Baselines**: Best baseline = [X] with [Y%] Good-or-better rate @K=[Z]
2. **Neural Models**: Best neural = [X] with [Y%] Good-or-better rate (improvement: [Δ%], p=[p-value])
3. **Symbolic Rules**: Rules [DID / DID NOT] improve metrics significantly (p=[p-value])
4. **Stability**: Stability policy [DID / DID NOT] reduce churn while maintaining performance
5. **Optimal K**: K*=[X] achieves [Y%] Good-or-better, [Z%] Unacceptable, Jaccard=[J]

## Statistical Significance
[Report all pairwise comparisons with p-values and confidence intervals]

## Interpretability Assessment
[Evaluate quality of rule evidence traces. Are explanations useful? Trustworthy?]

## Production Readiness
**Verdict**: [READY / NEEDS ITERATION / RESEARCH GAP]
**Justification**: [Evidence-based reasoning]
**Next Steps**: [If not ready, what needs to happen?]

## Lessons Learned
1. [Key insight 1]
2. [Key insight 2]
...

## Reproducibility Checklist
✅ All code, configs, and data processing scripts archived
✅ Random seeds documented
✅ Git commit hashes logged
✅ All artifacts generated and organized

---

**Project**: CA5 Neuro-Symbolic Predictive Maintenance
**Researcher**: y + Synapse
**Date**: [completion date]
```

---

## Success Criteria

✅ All variants evaluated with complete metrics
✅ Statistical significance tested (pairwise comparisons)
✅ Error analysis conducted (near-misses, failures)
✅ Research questions answered with evidence
✅ Production recommendation made with clear justification
✅ Comprehensive final report generated

---

## Next Steps (After Ablation)

- If production-ready → Create deployment plan
- If needs iteration → Identify specific improvements
- If research gap → Define follow-up experiments

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
