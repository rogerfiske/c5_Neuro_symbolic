# Project Memory: C5 Neuro-Symbolic Predictive Maintenance

## Project Overview
Neuro-symbolic AI system for predicting next-day staged parts pools for a 5-machine production line (CA5 project).

## Key Facts

### Dataset
- **File**: `data/raw/CA5_date.csv`
- **Records**: 11,685 daily part shipment records
- **Date Range**: 1992-02-04 to 2026-01-21 (34 years)
- **Format**: `date, m_1, m_2, m_3, m_4, m_5`
- **Parts**: IDs 1-39 (exactly 5 unique parts per day)

### Model Configuration
```yaml
encoder_type: transformer
embed_dim: 128
hidden_dim: 192
num_layers: 3
num_heads: 2
dropout: 0.2
sequence_length: 14  # 2 weeks (short context wins)
pool_size: 30
```

### Performance Metrics (same test split)
- **Good-or-Better (GoB)**: Primary metric (5/5 Excellent + 4/5 Good)
- **Hybrid @K=30**: 69.9% GoB (production)
- **Neural @K=30**: 68.2% GoB
- **Baseline @K=30**: 65.8% GoB
- **Neural Lift**: +2.4pp overall, +36.8pp on hard parts
- **Hybrid Lift**: +1.6pp over pure neural

### Part Categories
- **Hard Parts** (6): [8, 12, 13, 22, 23, 39] - Baseline <50% recall
- **Medium Parts** (20): Most parts
- **Easy Parts** (13): [2, 6, 9, 10, 11, 15, 17, 18, 19, 25, 26, 28, 29] - >80% recall

## Critical Findings

### K=39 Insight
At K=39 (all parts), any model achieves 100% accuracy trivially. Must evaluate at fixed K.

### Neural Value Concentration
Neural model excels on hard parts:
- Hard parts: +36.8pp lift over baseline
- Easy parts: +2.8pp lift over baseline

### Part 12 Anomaly - RESOLVED
- Neural: 0% recall (ranks 32-35, just outside K=30)
- Baseline: 36.5% recall
- **Root Cause**: Model assigns 0.004 lower probability, pushing it just outside pool
- **Solution**: Hybrid strategy (baseline for Part 12 only)

### Hybrid Strategy (Production Recommended)
- Neural for parts 1-11, 13-39
- Baseline for Part 12 ONLY
- Result: 69.9% GoB (+1.6pp over pure neural)

## Agent Activation
```
/synapse
```
Activates Dr. Synapse - Neuro-Symbolic ML Research Engineer

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Source CSV data |
| `scripts/` | Analysis scripts |
| `runpod_package/` | Deep learning pipeline for GPU |
| `outputs/` | Local training outputs |
| `Phase 2 outputs/` | RunPod B200 analysis results |

## Production Recommendation
**Deploy HYBRID strategy @K=30**
- Neural for parts 1-11, 13-39
- Baseline for Part 12 ONLY
- Use `python scripts/production_inference.py`
- Checkpoint: `outputs/best_model/checkpoints/`
- Fallback: frequency baseline

### Closed Research Directions
- **Column-enhanced neural** (2026-01-28): 6 architectures, none beat hybrid
- **Per-column frequency** (2026-01-28): Worse than global baseline
- **Ensemble strategies** (2026-01-26): 6 strategies, pure neural wins
- **Symbolic rules**: +0pp metric improvement, interpretability only

## Session History
- **2026-01-21**: Project initiated
- **2026-01-22**: Synapse agent created
- **2026-01-23**: RunPod H200 training (50 trials)
- **2026-01-26**: Phase 2 complete on B200 (per-part + ensemble analysis)
- **2026-01-27**: Part 12 investigation complete, hybrid strategy implemented
- **2026-01-28**: Column-enhanced research closed, no improvement over hybrid

## Research Status
All research phases complete. Production ready.

## Possible Future Directions
1. Monitoring dashboard (track live accuracy, detect drift)
2. Periodic retraining pipeline (as new data accumulates)
3. Dynamic K (vary pool size based on daily confidence)
4. Maintenance scheduling (regime detection -- PRD Section 8, never pursued)
5. Attention visualization (optional, for interpretability)
