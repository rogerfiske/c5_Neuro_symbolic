# Ensemble Experiment Results

Generated: 2026-01-26 19:33:40

Pool Size: K=30

## Strategy Comparison

| Strategy | Excellent | Good | GoB | Unacceptable | Avg Hits |
|----------|-----------|------|-----|--------------|----------|
| Pure Neural ** | 25.6% | 42.6% | 68.2% | 31.8% | 3.88 |
| Voting (25+25) | 24.9% | 42.2% | 67.1% | 32.9% | 3.86 |
| Confidence Weighted | 26.4% | 40.0% | 66.4% | 33.6% | 3.87 |
| Pure Baseline | 23.2% | 42.6% | 65.8% | 34.2% | 3.83 |
| Adaptive Hybrid | 23.3% | 41.9% | 65.2% | 34.8% | 3.80 |
| Hybrid (Neural Hard) | 23.3% | 41.6% | 64.9% | 35.1% | 3.81 |

## Analysis

- **Baseline GoB**: 65.8%
- **Neural GoB**: 68.2%
- **Neural Lift**: +2.5pp
- **Best Strategy**: Pure Neural (68.2%)
- **Best vs Baseline**: +2.5pp
- **Best vs Neural**: +0.0pp

## Recommendations

1. **Pure neural provides meaningful lift** over baseline
2. Ensemble provides marginal additional benefit
3. Consider pure neural for simplicity if compute allows