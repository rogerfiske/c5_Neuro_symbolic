# Neural Model Prototyping Workflow

**Workflow ID**: neural-model-prototype
**Purpose**: Build calibrated neural scorers with per-part probabilities
**Prerequisites**: feature-schema, rulebook-draft completed
**Estimated Duration**: 6-12 hours (local PC) OR 2-4 hours (RunPod H200)

---

## Objective

Train neural models to produce per-part scores/probabilities:
- Implement small prototypes (<5M params) on local PC first
- Scale to larger models (Transformer on 11K sequence) on H200 if justified
- Calibrate outputs to reflect true coverage likelihood
- Beat best baseline from baseline-suite workflow
- Prepare for neuro-symbolic integration (output logits + probabilities)

---

## Model Architectures to Explore

### 1. **Logistic Regression** (Baseline Neural)
- Input: Feature vector (recency + temporal + associations)
- Output: Per-part probabilities (39 logits → softmax)
- Pros: Fast, interpretable
- Cons: Linear, may not capture complex patterns

### 2. **GRU/LSTM** (Sequence Model)
- Input: Last W days as multi-hot sequence
- Output: Per-part logits (39)
- Pros: Captures temporal dependencies
- Cons: Medium complexity

### 3. **Transformer** (Full Sequence Model - H200)
- Input: Full 11K+ day sequence with positional encoding
- Output: Per-part logits (39)
- Pros: Long-range dependencies, state-of-the-art temporal modeling
- Cons: Expensive (requires H200 for large batch sizes)

### 4. **Temporal Point Process** (Advanced)
- Treat part occurrences as events with excitation/inhibition
- Hawkes process or Neural Hawkes
- Pros: Theoretically grounded for recurrence patterns
- Cons: Complex implementation

---

## Training Protocol

**Loss Function**: Multi-label binary cross-entropy (5 labels per day)
**Metrics**: Per-part AUC-ROC, calibration error (ECE)
**Early Stopping**: Patience=10 epochs, monitor validation tier rates
**Regularization**: Dropout, weight decay, label smoothing (optional)

**GPU Decision**:
- [ ] Local PC: Logistic, GRU/LSTM with W < 180 days
- [ ] RunPod H200: Transformer with full sequence, large batch sizes (128+)

---

## Calibration

**Why**: Raw neural scores ≠ true probabilities. Calibration critical for pool sizing.

**Methods**:
1. **Platt Scaling**: Logistic regression on validation scores
2. **Isotonic Regression**: Non-parametric monotonic transformation
3. **Temperature Scaling**: Global temperature parameter T

**Validation**: Plot reliability diagrams (predicted prob vs empirical frequency)

---

## Outputs

**Location**: `{project-root}/_bmad-output/synapse/neural-model-prototype/{run-id}/`

**Files**:
- `config.yaml` - Model hyperparameters, training config
- `metrics.csv` - Training/validation/test metrics (AUC, ECE, tier rates)
- `model_checkpoint.pt` - Saved model weights
- `calibration_curve.png` - Reliability diagram
- `tier_rates_vs_baseline.png` - Comparison with best baseline
- `training_log.md` - Loss curves, convergence analysis

---

## Success Criteria

✅ Model trained to convergence (early stopping triggered OR max epochs)
✅ Beats best baseline on Good-or-better rate @K (any K ∈ [20, 27])
✅ Calibration error (ECE) < 0.10
✅ Model checkpointed and reproducible
✅ All artifacts generated

---

## Next Workflow

**hybrid-inference** - Combine neural scores with symbolic rules

---

**Workflow Status**: Template Ready
**Last Updated**: 2026-01-22
