# Quick Reference Card - C5 Neuro-Symbolic Project

> **Keep this open during work sessions for fast lookups**

---

## 🚀 Quick Start

```bash
# Activate Synapse
/synapse

# View commands
MH

# Start profiling
DP

# Chat with agent
CH

# Exit
DA
```

---

## 🎯 Project Goals

**Predict**: Next-day parts pool (K=20-27)
**Maximize**: Good-or-better rate (≥4/5 in pool)
**Minimize**: Pool size K + Unacceptable rate (≤3/5)
**Maintain**: Stability (Jaccard ≥0.70)

---

## 📊 Dataset Quick Facts

- **File**: `data/raw/CA5_date.csv`
- **Records**: 11,685 (34 years: 1992-2026)
- **Format**: date, m_1, m_2, m_3, m_4, m_5
- **Parts**: 5 unique per day (IDs 1-39)

---

## 🧠 Synapse Commands

| Code | Command | Duration |
|------|---------|----------|
| **DP** | Data Profiling | 30min-2h |
| **BL** | Baseline Suite | 2-4h |
| **FS** | Feature Schema | 3-6h |
| **RD** | Rulebook Draft | 4-8h |
| **NP** | Neural Prototype | 6-12h |
| **HI** | Hybrid Inference | 4-8h |
| **KO** | K-Optimizer | 2-4h |
| **AR** | Ablation Report | 3-6h |

---

## 💻 Compute Resources

### Local PC (Use First)
- Feature engineering
- Baselines
- Small models (<1M params)
- Evaluation

### RunPod H200 (Use When)
- Transformers (11K+ sequence)
- Sweeps (>50 configs)
- Graph embeddings (>10K nodes)
- ILP rule mining

---

## ✅ Pre-Run Checklist (Critical!)

Before ANY training:
- [ ] Time-based split (no future leakage)
- [ ] 5 unique parts constraint enforced
- [ ] Tier metrics computed first
- [ ] Stability tracked (Jaccard + churn)
- [ ] Run logged (seed, config, git hash)
- [ ] Baseline comparison ready

---

## 📁 Key File Paths

### Agent
```
_bmad-output/bmb-creations/synapse/
├── synapse.agent.yaml
└── synapse-sidecar/
```

### Output
```
_bmad-output/synapse/{workflow}/{run-id}/
├── config.yaml
├── metrics.csv
├── run_log.md
└── plots/
```

### Docs
```
docs/prd_neurosymbolic_ai_ca5_v1_1.md
docs/bmad_agent_card_neurosymbolic_pm_ca5.md
Session_Summary_YYYY-MM-DD.md
Start_Here_Tomorrow_YYYY-MM-DD.md
```

---

## 🎓 Core Principles

1. **Baseline-First**: Beat simple models with evidence
2. **Reproducibility**: Seeds, configs, git hashes
3. **Interpretability**: Rule traces, counterfactuals
4. **No Leakage**: Rigorous feature audits
5. **Stability**: Avoid thrash unless justified

---

## 📈 Success Metrics

| Metric | Target |
|--------|--------|
| Good-or-better @K | ≥90% |
| Unacceptable @K | ≤5% |
| Pool size K* | Min [20,27] |
| Jaccard | ≥0.70 |
| Calibration ECE | <0.10 |

---

## 🔧 Common Commands

### Python Dependencies
```bash
pip install pandas matplotlib scipy numpy scikit-learn torch
```

### Check Agent Status
```bash
ls _bmad-output/bmb-creations/synapse/
```

### Review Workflow
```bash
cat _bmad-output/bmb-creations/synapse/synapse-sidecar/workflows/data-profile.md
```

---

## 🚨 Emergency Fixes

### Agent won't activate
1. Check validation passed
2. Verify installation location
3. Check sidecar paths

### Workflow file not found
1. Verify sidecar location
2. Check path variables: `{project-root}/_bmad/_memory/synapse-sidecar/`

### Missing dependencies
```bash
pip install pandas matplotlib scipy numpy scikit-learn
```

### Output folder error
```bash
mkdir -p _bmad-output/synapse
```

---

## 📝 Artifact Checklist

Every workflow must produce:
- [ ] config.yaml (parameters, seeds)
- [ ] metrics.csv (results)
- [ ] run_log.md (execution trace)
- [ ] plots/ (visualizations)

---

## 🎯 Today's Focus (2026-01-23)

**Primary**:
1. Run validation ([V] option)
2. Install Synapse
3. Execute data profiling (DP)

**Secondary**:
4. Start baseline suite (BL)

**Success**: Dataset profiled + baselines started

---

## 🔗 Quick Links

- **PRD**: `docs/prd_neurosymbolic_ai_ca5_v1_1.md`
- **Checklist**: `_bmad-output/bmb-creations/synapse/synapse-sidecar/pre-run-checklist.md`
- **Instructions**: `_bmad-output/bmb-creations/synapse/synapse-sidecar/instructions.md`

---

## 💡 Remember

- Start simple (baselines first)
- Log everything (reproducibility)
- No leakage (audit features)
- Stability matters (Jaccard penalty)
- Trust the checklist (prevents 90% of failures)

---

**Reference Card Version**: 1.0
**Last Updated**: 2026-01-22
**Print-friendly**: Yes
