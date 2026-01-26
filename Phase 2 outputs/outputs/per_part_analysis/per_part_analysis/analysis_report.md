# Per-Part Neural vs Baseline Analysis

Generated: 2026-01-26 19:33:31

Pool Size: K=30

## Summary by Part Category

| Category | Parts | Occurrences | Neural Recall | Baseline Recall | Neural Lift | Neural Wins | Baseline Wins | Net Advantage |
|----------|-------|-------------|---------------|-----------------|-------------|-------------|---------------|---------------|
| HARD | 6 | 486 | 84.8% | 47.9% | +36.8pp | 206 | 27 | +179 |
| MEDIUM | 20 | 1881 | 85.9% | 56.3% | +29.6pp | 697 | 141 | +556 |
| EASY | 13 | 1283 | 62.6% | 59.8% | +2.8pp | 316 | 280 | +36 |
| **TOTAL** | 39 | 3650 | 77.5% | 56.4% | +21.1pp | 1219 | 448 | +771 |

## Key Findings

**CONFIRMED: Neural model provides greater lift on HARD parts.**

- Hard parts neural lift: +36.8pp
- Easy parts neural lift: +2.8pp
- Differential: 34.0pp more lift on hard parts

## Per-Part Breakdown (Hard Parts)

| Part ID | Occurrences | Neural Recall | Baseline Recall | Lift |
|---------|-------------|---------------|-----------------|------|
| 12 | 74 | 0.0% | 36.5% | -36.5pp |
| 8 | 78 | 100.0% | 48.7% | +51.3pp |
| 13 | 83 | 100.0% | 42.2% | +57.8pp |
| 22 | 78 | 100.0% | 50.0% | +50.0pp |
| 23 | 80 | 100.0% | 52.5% | +47.5pp |
| 39 | 93 | 100.0% | 55.9% | +44.1pp |

## Recommendations

1. **Ensemble Strategy Viable**: Neural model adds meaningful value on hard parts
2. **Proposed Ensemble**: Use neural predictions for hard parts, baseline for easy parts
3. **Expected Benefit**: Reduced compute while maintaining neural advantage where it matters