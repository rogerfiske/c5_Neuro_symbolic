# Start Here Tomorrow: 2026-01-29

## /synapse

## Quick Context
**Project**: C5 Neuro-Symbolic Predictive Maintenance
**Status**: Production Ready - All Research Complete, Inference Working Locally
**Last Session**: Doc cleanup, inference fix, first prediction generated

---

## What Was Accomplished (2026-01-28)

### Documentation Cleanup
- `.project_memory.md` fully rewritten (was 5 sessions stale)
- `README.md` contradictions fixed (hybrid is the recommendation, not pure neural)
- `CLAUDE.md` updated with closed research directions and current status

### Production Inference Fixed
- Checkpoint path corrected (`outputs/outputs/best_model/`)
- NumPy 1.x/2.x pickle incompatibility resolved (re-saved as pure torch tensors)
- Neural model now loads and runs locally

### New: Predict Command
- `PR` added to Synapse menu
- Runs hybrid inference, saves to `predictions/` directory

---

## Quick Commands

### Run Daily Prediction
```bash
cd C:\Users\Minis\CascadeProjects\c5_neuro_symbolic
python scripts/production_inference.py --date 2026-01-29 --output predictions/prediction_2026-01-29.txt
```

### Or via Synapse Menu
```
/synapse
PR
```

### Baseline-Only Mode (No Neural Model)
```bash
python scripts/production_inference.py --baseline-only --output predictions/prediction_2026-01-29.txt
```

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

## Synapse Menu

| Code | Workflow |
|------|----------|
| DP | Data Profiling |
| BL | Baseline Suite |
| FS | Feature Schema |
| RD | Rulebook Draft |
| NP | Neural Prototype |
| HI | Hybrid Inference |
| KO | K-Optimizer |
| AR | Ablation Report |
| **PR** | **Predict (NEW)** |

---

## Project Status

| Phase | Status |
|-------|--------|
| Phase 1: Baseline & Neural | COMPLETE |
| Phase 2: Per-Part & Ensemble | COMPLETE |
| Part 12 Investigation | COMPLETE |
| Hybrid Strategy | IMPLEMENTED |
| Column-Enhanced Research | COMPLETE (No Improvement) |
| Documentation Cleanup | COMPLETE |
| Production Inference | WORKING LOCALLY |
| **First Prediction** | **GENERATED (2026-01-28)** |

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/production_inference.py` | Production inference (hybrid strategy) |
| `outputs/outputs/best_model/checkpoints/model_state_dict.pt` | Neural model weights |
| `outputs/outputs/best_model/config.yaml` | Model configuration |
| `predictions/` | Saved prediction outputs |

---

## Optional Future Work

1. **Score predictions against actuals** -- when today's real parts are known, compare against yesterday's prediction
2. **Monitoring dashboard** -- track tier rates over time
3. **Periodic retraining** -- update model as new data accumulates
4. **Dynamic K** -- vary pool size based on daily confidence
5. **Maintenance scheduling** -- regime detection (PRD Section 8, never pursued)

---

## GitHub Repo
https://github.com/rogerfiske/c5_Neuro_symbolic.git
