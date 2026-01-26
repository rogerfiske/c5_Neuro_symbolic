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

### Performance Metrics
- **Good-or-Better (GoB)**: Primary metric (5/5 Excellent + 4/5 Good)
- **Baseline @K=30**: 65.8% GoB
- **Neural @K=30**: 68.2% GoB
- **Neural Lift**: +2.5pp overall, +36.8pp on hard parts

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

### Part 12 Anomaly
- Neural: 0% recall
- Baseline: 36.5% recall
- **Investigation needed** - script at `scripts/part12_investigation.py`

### Ensemble Finding
Pure neural is optimal. All ensemble strategies tested underperform pure neural.

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
**Deploy pure neural model @K=30**
- Use checkpoint from `outputs/best_model/checkpoints/`
- Consider special handling for Part 12
- Fallback: frequency baseline

## Session History
- **2026-01-21**: Project initiated
- **2026-01-22**: Synapse agent created
- **2026-01-23**: RunPod H200 training (50 trials)
- **2026-01-26**: Phase 2 complete on B200 (per-part + ensemble analysis)

## Next Steps
1. Investigate Part 12 anomaly
2. Attention analysis
3. Production deployment preparation
