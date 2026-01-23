# Pre-Run Checklist Template

**CRITICAL**: Complete this checklist BEFORE starting ANY model training or experiment.

---

## Experiment Metadata

- **Run ID**: `_____________________` (UUID or timestamp)
- **Workflow**: `_____________________` (baseline-suite, neural-model-prototype, etc.)
- **Date**: `_____________________`
- **Researcher**: y
- **Git Commit Hash**: `_____________________`

---

## 1. Data Integrity & Leakage Prevention

- [ ] **Time-based split confirmed**: Train/validation/test splits respect temporal ordering
- [ ] **No future leakage**: Features computed ONLY from past data (no peeking ahead)
- [ ] **Day-level constraint verified**: Training data contains exactly 5 unique parts per day
- [ ] **Prediction constraint enforced**: Model outputs exactly 5 unique parts (or pool of K unique parts)

**Notes**:
```
[Document split strategy, feature construction windows, validation approach]
```

---

## 2. Baseline Comparison

- [ ] **Baseline models identified**: List baseline approaches to beat
  - [ ] Frequency-based ranking
  - [ ] Recency-based ranking
  - [ ] Co-occurrence heuristics
  - [ ] Other: `__________________`
- [ ] **Baseline results documented**: Tier metrics (Excellent/Good/Unacceptable rates) recorded
- [ ] **Complexity advancement justified**: Current approach beats baselines OR provides interpretability gains

**Baseline Performance**:
```
Baseline: [name]
- Excellent rate @K=[X]: ____%
- Good-or-better rate @K=[X]: ____%
- Unacceptable rate @K=[X]: ____%
```

---

## 3. Tier Metrics & Service Levels

- [ ] **Primary metrics defined**: Excellent rate, Good-or-better rate, Unacceptable rate
- [ ] **Tier computation order**: Tier metrics computed BEFORE secondary metrics (calibration, stability)
- [ ] **K range specified**: Pool size K ∈ [20, 27] (or documented alternative)
- [ ] **Unacceptable ceiling set**: Maximum tolerable unacceptable rate = `_____%`

**Target Service Levels**:
```
- Good-or-better rate: ≥ _____%
- Unacceptable rate: ≤ _____%
- Preferred K: _____ (within [20,27])
```

---

## 4. Stability & Pool Churn

- [ ] **Stability metric defined**: Jaccard similarity or other measure
- [ ] **Churn tracking enabled**: Day-to-day pool changes logged
- [ ] **Thrash detection configured**: Flag excessive instability (Jaccard < threshold)
- [ ] **Strong-shift criteria defined**: Conditions justifying significant pool changes

**Stability Thresholds**:
```
- Target Jaccard similarity: ≥ _____
- Thrash alert threshold: Jaccard < _____
- Strong-shift triggers: [document conditions]
```

---

## 5. Reproducibility Requirements

- [ ] **Random seed set**: Python, NumPy, PyTorch, etc. all seeded
- [ ] **Config file prepared**: YAML/JSON with ALL hyperparameters
- [ ] **Run ID generated**: Unique identifier for this experiment
- [ ] **Git hash logged**: Current commit hash recorded
- [ ] **Environment documented**: Python version, library versions, GPU specs (if applicable)

**Configuration Snapshot**:
```yaml
seed: _____
window_size: _____
learning_rate: _____
batch_size: _____
max_epochs: _____
# [Add all relevant hyperparameters]
```

---

## 6. Code Quality & Safety

- [ ] **Early stopping configured**: Max iterations/epochs + patience parameter
- [ ] **Progress monitoring enabled**: Terminal output for runs >10 minutes (progress bars, ETA)
- [ ] **Infinite loop detection**: Hard iteration counters + timeout guards
- [ ] **Exception handling**: Informative error messages + graceful degradation
- [ ] **Checkpointing enabled**: Periodic saves (every N iterations) with auto-resume

**Safety Configuration**:
```
- Max iterations: _____
- Patience (early stopping): _____ epochs
- Timeout (wall-clock): _____ hours
- Checkpoint frequency: every _____ iterations
```

---

## 7. GPU Decision Logic

- [ ] **Compute requirements estimated**: Model size, sequence length, batch size
- [ ] **Resource allocation decided**: Local PC or RunPod H200
- [ ] **Justification documented**: Why this resource choice

**Resource Decision**:
- [ ] **Local PC** (Feature eng, baselines, small prototypes <1M params, evaluation)
- [ ] **RunPod H200** (Large transformers, sweeps >50 configs, graph embeddings >10K nodes, ILP mining)

**Justification**:
```
[Why this resource choice makes sense for this experiment]
```

---

## 8. Artifact Generation Plan

- [ ] **Output folder created**: `{project-root}/_bmad-output/synapse/{workflow}/{run-id}/`
- [ ] **Config file**: Will save to `config.yaml`
- [ ] **Metrics file**: Will save to `metrics.csv` (or JSON)
- [ ] **Log file**: Will save to `run_log.md`
- [ ] **Visualizations**: Will save key plots (tier rates, calibration curves, stability over time)

**Expected Artifacts**:
```
- config.yaml (hyperparameters, seeds, timestamps)
- metrics.csv (Excellent/Good/Unacceptable rates, Jaccard, etc.)
- run_log.md (execution trace, errors, warnings)
- plots/ (tier_rates.png, calibration_curve.png, stability.png)
```

---

## 9. Interpretability & Explainability

- [ ] **Rule evidence plan**: How will symbolic rules be extracted/validated?
- [ ] **Counterfactual checks**: Plan for testing "what-if" scenarios
- [ ] **Decision trace logging**: Record which rules/features drove predictions

**Interpretability Strategy**:
```
[Describe how explanations will be generated and validated]
```

---

## 10. Final Sign-Off

- [ ] **All checklist items completed**
- [ ] **Baseline comparison strategy clear**
- [ ] **Reproducibility safeguards in place**
- [ ] **Artifact generation plan confirmed**

**Ready to proceed**: YES / NO

**Researcher Sign-Off**: y
**Date**: `_____________________`

---

## Post-Run Actions

After experiment completes:
- [ ] Save all artifacts to output folder
- [ ] Update baseline performance table if new best
- [ ] Document lessons learned
- [ ] Archive config + metrics for future reference

---

**CRITICAL REMINDER**: If any checklist item is NO or incomplete, DO NOT proceed with training. Fix the gap first. Research without rigor is research theater.
