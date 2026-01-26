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
