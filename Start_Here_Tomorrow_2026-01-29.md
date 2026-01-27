# Start Here Tomorrow: 2026-01-29

## /synapse

## Quick Context
**Project**: C5 Neuro-Symbolic Predictive Maintenance
**Status**: Production Ready (Hybrid Strategy) - All Research Complete
**Last Session**: Column-enhanced experiments completed, no improvement found

---

## What Was Accomplished (2026-01-28)

### Column-Enhanced Neural Experiments (RunPod B200)

Tested 6 neural architectures incorporating column-position information:

| Configuration | GoB | vs Hybrid |
|---------------|-----|-----------|
| column_output_heads | 68.08% | -1.82pp |
| baseline_standard | 67.67% | -2.23pp |
| column_aware_with_heads | 67.40% | -2.50pp |
| column_features_embed | 67.26% | -2.64pp |
| column_aware_embed | 66.58% | -3.32pp |
| column_features_with_heads | 66.44% | -3.46pp |

**Result**: None beat the hybrid strategy (69.9% GoB). Research direction CLOSED.

### Per-Column Frequency Baseline

| Strategy | GoB | vs Global |
|----------|-----|-----------|
| Global Baseline | 68.67% | - |
| Per-Col Optimized K | 66.89% | -1.78pp |

**Result**: Per-column frequency also did NOT improve. Research direction CLOSED.

---

## Final Production Strategy

| Metric | Hybrid (Recommended) |
|--------|----------------------|
| Excellent | 27.0% |
| Good | 42.9% |
| **GoB** | **69.9%** |
| Unacceptable | 30.1% |

**Approach**: Neural for parts 1-11, 13-39 + Baseline for Part 12 only

---

## Quick Commands

### Run Production Inference (Hybrid)
```bash
cd C:\Users\Minis\CascadeProjects\c5_neuro_symbolic
python scripts/production_inference.py
```

### Predict for Specific Date
```bash
python scripts/production_inference.py --date 2026-01-30
```

### Baseline-Only Mode (No GPU)
```bash
python scripts/production_inference.py --baseline-only
```

---

## Project Status Summary

| Phase | Status |
|-------|--------|
| Phase 1: Baseline & Neural | COMPLETE |
| Phase 2: Per-Part & Ensemble | COMPLETE |
| Part 12 Investigation | COMPLETE |
| Hybrid Strategy | IMPLEMENTED |
| Column-Enhanced Research | COMPLETE (No Improvement) |
| **Production Inference** | **READY** |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/production_inference.py` | Production inference (hybrid strategy) |
| `outputs/best_model/checkpoints/` | Neural model checkpoint |
| `docs/research_backlog_neural_column_features.md` | Column research (closed) |

---

## Optional Future Work

1. **Monitoring Dashboard** - Track actual tier rates vs predictions
2. **Model Retraining** - Periodic retraining as new data accumulates
3. **Attention Analysis** - Visualize what the model learned

---

## GitHub Repo
https://github.com/rogerfiske/c5_Neuro_symbolic.git
