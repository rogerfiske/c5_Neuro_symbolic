# RunPod Phase 2 Analysis Package

## Quick Start (B200 with PyTorch 2.8.0)

```bash
# 1. Upload and extract package
cd /workspace
unzip runpod_package.zip
cd runpod_package

# 2. Install dependencies (most already in template)
pip install -r requirements.txt

# 3. Copy data and previous outputs
# Ensure data/CA5_date.csv exists
# Ensure outputs/best_model/checkpoints/ has model checkpoints

# 4. Run Phase 2 analysis (both scripts in sequence)
python run_phase2_analysis.py

# Or run individually:
python per_part_inference.py      # Per-part accuracy analysis
python ensemble_experiment.py     # Ensemble strategy comparison
```

## Output Files

After running, download these from `outputs/`:

```
outputs/
├── per_part_analysis/
│   ├── per_part_results.csv           # Every prediction by part
│   ├── neural_vs_baseline_by_category.csv  # Summary by hard/medium/easy
│   └── analysis_report.md             # Markdown report
└── ensemble_experiment/
    ├── ensemble_results.csv           # Strategy comparison
    └── ensemble_report.md             # Recommendations
```

## Expected Results

Based on Phase 1 findings:
- **Neural recall @K=30**: ~72.4% GoB (validation), ~69% GoB (test)
- **Baseline recall @K=30**: ~68.9% GoB
- **Neural lift**: ~+3.5pp

Phase 2 will determine:
1. Is neural lift concentrated on hard parts?
2. Can ensemble strategies exceed pure neural?
3. Recommended production approach

## Hardware Compatibility

Tested on:
- RunPod H200 (141GB HBM3) with PyTorch 2.4.x
- RunPod B200 with PyTorch 2.8.0

The B200 with PyTorch 2.8.0 may offer:
- Better torch.compile support
- Improved memory efficiency
- Faster attention computation

---

## Phase 3: Column-Enhanced Neural Models (2026-01-27)

### Overview

Tests neural architectures that incorporate column-position information.
Each column (m_1 to m_5) has distinct part distributions - the neural model
should learn to exploit these patterns.

### Configurations Tested

| Name | Embedding | Output Heads | Description |
|------|-----------|--------------|-------------|
| baseline_standard | Standard | Single | Original architecture |
| column_aware_embed | Column-Aware | Single | Adds column embeddings |
| column_features_embed | Column-Features | Single | Adds explicit column stats |
| column_output_heads | Standard | Per-Column | 5 separate prediction heads |
| column_aware_with_heads | Column-Aware | Per-Column | Combined approach |
| column_features_with_heads | Column-Features | Per-Column | Full enhancement |

### Quick Start

```bash
# Run column-enhanced experiments
python train_column_enhanced.py

# Results saved to:
# outputs/column_enhanced/experiment_results.csv
# outputs/column_enhanced/{config_name}/checkpoints/
```

### Expected Improvements

Based on column distribution analysis:
- m_1 concentrates on parts 1-18
- m_5 concentrates on parts 22-39
- Parts like Part 12 may benefit from column-specific handling

Target: >70% GoB (vs current 68.2% pure neural, 69.9% hybrid)
