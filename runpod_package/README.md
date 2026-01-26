  # Neuro-Symbolic Part Prediction Pipeline

Deep learning pipeline for CA5 part prediction combining neural networks with symbolic rules.

## Requirements

- RunPod template: **PyTorch 2.4.0** (or later)
- GPU: H200 recommended (will work on any CUDA GPU)
- Storage: ~2GB for data, models, and outputs

## Quick Start

### 1. Upload to RunPod

1. Zip this entire folder
2. Upload to RunPod pod via Jupyter or SSH
3. Extract to `/workspace/neuro_symbolic/`

### 2. Install Dependencies

```bash
cd /workspace/neuro_symbolic
pip install -r requirements.txt
```

### 3. Run Training

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter lab
# Open neuro_symbolic_pipeline.ipynb
```

**Option B: Command Line**
```bash
# Quick test (5 epochs)
python train.py --max_epochs 5 --batch_size 64

# Full training
python train.py --max_epochs 100 --encoder_type transformer

# Hyperparameter optimization
python hyperopt.py --n_trials 50 --timeout 2.0 --train_best
```

## File Structure

```
runpod_package/
├── data/
│   └── CA5_date.csv          # Input data
├── models/
│   ├── __init__.py
│   └── neuro_symbolic.py     # Model architecture
├── outputs/                   # Results go here
├── data_module.py            # Data loading
├── train.py                  # Training script
├── hyperopt.py               # Hyperparameter optimization
├── neuro_symbolic_pipeline.ipynb  # Interactive notebook
├── requirements.txt
└── README.md
```

## Model Architecture

```
Input: 30-day sequence of part usage (30 x 5 parts)
    ↓
Part Embedding: Learned representations (64-dim)
    ↓
Temporal Encoder: Transformer or LSTM (2-4 layers)
    ↓
Context Aggregation: Pool temporal features
    ↓
┌──────────────────────────────────────────┐
│         Neuro-Symbolic Fusion            │
├──────────────────────────────────────────┤
│  Neural Head ──┐                         │
│                ├── Fusion Gate ──→ Logits│
│  Symbolic Attn─┘                         │
└──────────────────────────────────────────┘
    ↓
Output: 39 per-part probabilities → Top-K pool
```

## Hyperparameters

Key parameters to tune:

| Parameter | Range | Default |
|-----------|-------|---------|
| encoder_type | lstm, transformer | transformer |
| embed_dim | 32-128 | 64 |
| hidden_dim | 64-256 | 128 |
| num_layers | 1-4 | 2 |
| learning_rate | 1e-5 to 1e-2 | 1e-3 |
| sequence_length | 14-60 | 30 |
| pool_size | 20-30 | 27 |
| dropout | 0-0.3 | 0.1 |

## Expected Results

Baseline (Last-30-Days): **53.1%** Good-or-Better

Target: Beat baseline with neural + symbolic combination

## Outputs

After training:
- `outputs/checkpoints/` - Model checkpoints
- `outputs/logs/` - TensorBoard logs
- `outputs/hyperopt/` - Optimization results
  - `best_params.yaml` - Best hyperparameters
  - `optimization_history.html` - Interactive plot
  - `param_importance.html` - Feature importance

## Monitoring

TensorBoard:
```bash
tensorboard --logdir outputs/logs --bind_all
```

## Troubleshooting

**Out of memory:**
- Reduce `batch_size`
- Reduce `hidden_dim` or `embed_dim`
- Use `precision='bf16-mixed'` instead of `'16-mixed'`

**Slow training:**
- Increase `batch_size` (H200 can handle 256+)
- Reduce `sequence_length`
- Use fewer `num_workers` if I/O bound

**Poor results:**
- Try longer `sequence_length` (45, 60)
- Enable `use_symbolic_init: True`
- Run more hyperopt trials

## Version History

### v1.1 (2026-01-23)
- Fixed SymbolicAttention tensor broadcasting (einsum → matmul)
- Fixed Rich/Jupyter recursion (TQDMProgressBar)
- Fixed PyTorch 2.6 weights_only compatibility
- Made optuna-integration imports optional
- Verified working on RunPod H200 through Step 5

### v1.0 (2026-01-23)
- Initial release

---
*Dr. Synapse Research Pipeline*
