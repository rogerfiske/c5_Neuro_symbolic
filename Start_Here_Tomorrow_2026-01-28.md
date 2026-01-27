# Start Here Tomorrow: 2026-01-28

## /synapse

## Quick Context
**Project**: C5 Neuro-Symbolic Predictive Maintenance
**Status**: Production Ready (Hybrid Strategy)
**Last Session**: Part 12 investigation complete, hybrid strategy implemented

---

## What Was Accomplished Yesterday (2026-01-27)

1. **Part 12 Investigation Complete**
   - Root cause: Part 12 ranks 32-35 (just outside K=30) on every prediction
   - Only 0.004 probability gap from parts that make the cutoff
   - Not a data rarity issue (rank 27/39 in frequency)
   - Declining trend (14.8% in 2017 -> 9.3% in 2025) may have influenced model

2. **Hybrid Strategy Implemented**
   - Neural for parts 1-11, 13-39
   - Baseline for Part 12 ONLY
   - Result: **69.9% GoB** (+1.6pp over pure neural 68.2%)
   - 12 days improved, 0 days degraded

3. **Production Inference Script Created**
   - `scripts/production_inference.py`
   - Supports hybrid and baseline-only modes
   - Ready for deployment

---

## Current Production Strategy

| Metric | Pure Neural | Hybrid (Recommended) |
|--------|-------------|----------------------|
| Excellent | 25.6% | 27.0% |
| Good | 42.6% | 42.9% |
| **GoB** | 68.2% | **69.9%** |
| Unacceptable | 31.8% | 30.1% |

---

## Quick Commands

### Run Production Inference (Hybrid)
```bash
cd C:\Users\Minis\CascadeProjects\c5_neuro_symbolic
python scripts/production_inference.py
```

### Predict for Specific Date
```bash
python scripts/production_inference.py --date 2026-01-29
```

### Baseline-Only Mode (No GPU)
```bash
python scripts/production_inference.py --baseline-only
```

---

## Optional Next Steps

### 1. Attention Analysis (Nice to Have)
Understand what the Transformer "sees" that makes it undervalue Part 12
- Would provide deeper model interpretability
- Not required for production

### 2. Monitoring Dashboard
- Track actual tier rates vs predictions
- Alert if model performance degrades

### 3. Model Retraining Schedule
- Consider periodic retraining as new data accumulates
- Part 12 trend (declining frequency) may continue

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/production_inference.py` | Production inference (hybrid strategy) |
| `scripts/part12_investigation.py` | Part 12 analysis (already run) |
| `scripts/part12_hybrid_evaluation.py` | Hybrid strategy evaluation |
| `outputs/part12_hybrid_evaluation.csv` | Daily hybrid results |
| `outputs/best_model/checkpoints/` | Neural model checkpoint |

---

## Project Status Summary

| Phase | Status |
|-------|--------|
| Phase 1: Baseline & Neural | COMPLETE |
| Phase 2: Per-Part & Ensemble | COMPLETE |
| Part 12 Investigation | COMPLETE |
| Hybrid Strategy | IMPLEMENTED |
| Production Inference | READY |

---

## GitHub Repo
https://github.com/rogerfiske/c5_Neuro_symbolic.git

**Note**: Commit today's changes before continuing.
