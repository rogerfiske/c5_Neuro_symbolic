# C5 Neuro-Symbolic Predictive Maintenance

**Project Status**: Agent Validation In Progress 🔄 | Research Phase Pending ⏳
**Last Updated**: 2026-01-23
**Researcher**: y
**Repository**: https://github.com/rogerfiske/c5_Neuro_symbolic.git

---

## Project Overview

Neuro-symbolic AI approach for predicting next-day staged parts pools for a 5-machine production line (CA5 project). Optimizes pool size (K=20-27) while enforcing tiered service constraints.

### Service Level Definitions
- **5/5 covered** = Excellent
- **4/5 covered** = Good
- **≤3/5 covered** = Unacceptable

### Key Objectives
1. **Maximize Good-or-better rate** (≥4 of 5 true parts in pool)
2. **Minimize pool size K** (inventory cost proxy)
3. **Maintain stability** (low daily churn)
4. **Provide interpretability** (rule evidence traces)

---

## Dataset

**File**: `data/raw/CA5_date.csv`
- **Records**: 11,685 daily part shipment records
- **Date Range**: 1992-02-04 to 2026-01-21 (34 years)
- **Format**: `date, m_1, m_2, m_3, m_4, m_5`
- **Part Domain**: IDs 1-39 (exactly 5 unique parts per day)
- **Invariants**: No duplicates within a day

### Known Characteristics
- Near-uniform global part frequencies
- Calendar gaps (early weekends, COVID period irregularities)
- 5 identical machines running 18 hours/day since 1992

---

## Technical Approach

### Two-Tier Neuro-Symbolic Architecture

**Tier A (Neural)**: Learn per-part scores/probabilities from temporal patterns
- Baseline models (frequency, recency, co-occurrence)
- Neural scorers (Logistic, GRU/LSTM, Transformer)
- Calibrated outputs for reliable pool sizing

**Tier B (Symbolic)**: Apply rules, constraints, stability policies
- Symbolic rule discovery (cooldowns, co-occurrence, bursts)
- Hard constraint enforcement (5 parts needed, K unique)
- Stability optimization (Jaccard penalty for churn)

---

## Project Structure

```
c5_neuro_symbolic/
├── data/
│   └── raw/
│       └── CA5_date.csv                 # Source dataset
├── docs/
│   ├── prd_neurosymbolic_ai_ca5_v1_1.md # Enhanced PRD
│   ├── bmad_agent_card_neurosymbolic_pm_ca5.md
│   ├── handoff_synapse_agent_to_builder.md
│   └── pc_specs.md
├── _bmad-output/
│   ├── bmb-creations/
│   │   └── synapse/                     # Agent build output
│   │       ├── synapse.agent.yaml
│   │       └── synapse-sidecar/         # 12 files
│   └── synapse/                         # Research workflow outputs
│       ├── data-profile/{run-id}/
│       ├── baseline-suite/{run-id}/
│       ├── feature-schema/{run-id}/
│       ├── rulebook-draft/{run-id}/
│       ├── neural-model-prototype/{run-id}/
│       ├── hybrid-inference/{run-id}/
│       ├── k-optimizer/{run-id}/
│       └── ablation-report/{run-id}/
├── Session_Summary_2026-01-22.md        # Daily progress log
├── Start_Here_Tomorrow_2026-01-23.md    # Next session guide
└── README.md                            # This file
```

---

## Synapse Agent

**Name**: Synapse
**Type**: Expert BMAD Agent
**Icon**: 🧠
**Status**: Built ✅ | Validated ⏳ | Deployed ⏳

### Agent Capabilities

**8 Research Workflows**:
1. **[DP] Data Profiling** - Validate schema, detect gaps, analyze distributions
2. **[BL] Baseline Suite** - Build strong baselines with tier metrics
3. **[FS] Feature Schema** - Engineer features with leakage audits
4. **[RD] Rulebook Draft** - Discover symbolic rules with evidence
5. **[NP] Neural Prototype** - Build calibrated neural scorers
6. **[HI] Hybrid Inference** - Combine neural + symbolic reasoning
7. **[KO] K-Optimizer** - Choose optimal pool size under constraints
8. **[AR] Ablation Report** - Systematic comparison with conclusions

### Agent Personality

- **Methodical researcher** with baseline-first rigor
- Refuses to advance complexity without evidence
- Treats reproducibility as non-negotiable standard
- Views interpretability as engineering requirement

### Sidecar Contents
- `instructions.md` - Core execution protocols
- `prd.md` - Complete enhanced PRD
- `pre-run-checklist.md` - 10-section research rigor template
- `workflows/` - 8 detailed workflow files with code examples

---

## Computational Resources

### Local PC
- **CPU**: AMD Ryzen 9 6900HX
- **RAM**: 64GB @ 2393MHz
- **GPU**: AMD Radeon RX 6600M (8GB VRAM)
- **Use For**: Feature engineering, baselines, small prototypes (<1M params), evaluation

### RunPod H200
- **GPU**: NVIDIA H200 (141GB HBM3)
- **Use For**: Large transformers (11K+ sequence), hyperparameter sweeps (>50 configs), graph embeddings (>10K nodes), ILP rule mining

---

## Research Workflow Progression

### Phase 0: Foundation (Current)
- [x] Dataset obtained (CA5_date.csv)
- [x] PRD created and enhanced
- [x] Agent card defined
- [x] Synapse agent built
- [~] **Agent validation in progress** (step 1 of 6 complete) ← CURRENT
- [ ] Agent installed and activated

### Phase 1: Data Understanding
- [ ] Data profiling (gaps, distributions, drift)
- [ ] Baseline suite (frequency, recency, co-occurrence)
- [ ] Performance benchmarks established

### Phase 2: Feature Engineering
- [ ] Multi-hot encoding
- [ ] Recency features (TSLU)
- [ ] Temporal features (day-of-week, gaps)
- [ ] Association features (co-occurrence)
- [ ] Leakage audit passed

### Phase 3: Symbolic Rules
- [ ] Association rule mining (Apriori, FP-Growth)
- [ ] Temporal pattern discovery
- [ ] Cooldown/burst rules
- [ ] Rule validation (support, confidence, lift)

### Phase 4: Neural Models
- [ ] Logistic regression baseline
- [ ] GRU/LSTM sequence models
- [ ] Transformer (full sequence on H200)
- [ ] Calibration (Platt, isotonic, temperature)

### Phase 5: Neuro-Symbolic Integration
- [ ] Score adjustment via rules
- [ ] Hard constraint enforcement
- [ ] Stability policy (Jaccard penalty)
- [ ] Rule evidence traces

### Phase 6: Optimization
- [ ] K optimization (constrained or cost-based)
- [ ] Tier rate analysis (K ∈ [20, 27])
- [ ] Sensitivity analysis
- [ ] Final recommendation

### Phase 7: Evaluation
- [ ] Ablation study (all variants)
- [ ] Statistical significance testing
- [ ] Error analysis (near-misses, failures)
- [ ] Research conclusion

---

## Key Principles (Enforced by Synapse)

1. **Baseline-First**: Start simple, advance with evidence
2. **Constraints are First-Class**: Encode symbolically, audit rigorously
3. **Reproducibility is Non-Negotiable**: Seeds, configs, git hashes logged
4. **Tiered Optimization**: Maximize Good-or-better, minimize K, minimize unacceptable
5. **Stability Matters**: Avoid thrash unless confidence justifies
6. **Interpretability is Engineering**: Rule evidence, counterfactuals, decision traces

---

## Code Quality Standards

ALL scripts must include:
- ✅ Early stopping (configurable max iterations)
- ✅ Progress monitoring (terminal output for runs >10 min)
- ✅ Infinite loop detection (counters + timeouts)
- ✅ Reproducibility (seed logging, config snapshots, run IDs)
- ✅ Exception handling (informative errors)

---

## Artifact Standards

Every workflow execution produces:
- **config.yaml** - Parameters, seeds, timestamps
- **metrics.csv** - Evaluation results
- **run_log.md** - Execution trace
- **visualizations/** - Key plots (PNG/HTML)

Output location: `_bmad-output/synapse/{workflow}/{run-id}/`

---

## Definition of Done (Research Stage)

A workflow is COMPLETE when:
- ✅ Achieves materially improved **Good-or-better rate** vs baselines within K ∈ [20, 27]
- ✅ Keeps **Unacceptable rate** below agreed ceiling
- ✅ Demonstrates stable pools (reasonable churn) except when "strong shift" triggers
- ✅ Provides interpretable rule evidence and reproducible experiment logs
- ✅ Produces artifacts: config, metrics, logs, plots

---

## Quick Start (After Validation)

1. **Activate Synapse**: `/synapse`
2. **View commands**: `MH` or `help`
3. **Start profiling**: `DP` or `data-profile`
4. **Chat with agent**: `CH` or `chat`
5. **Exit agent**: `DA` or `exit`

---

## Dependencies

### Python Packages
```bash
pip install pandas matplotlib scipy numpy scikit-learn torch
```

### Optional (Advanced)
```bash
pip install pymc stan problog  # Probabilistic programming
pip install transformers        # Hugging Face models
pip install networkx            # Graph analysis
```

---

## Documentation

### Primary Documents
- **PRD**: `docs/prd_neurosymbolic_ai_ca5_v1_1.md`
- **Agent Card**: `docs/bmad_agent_card_neurosymbolic_pm_ca5.md`
- **Handoff Doc**: `docs/handoff_synapse_agent_to_builder.md`

### Session Logs
- **Session Summary**: `Session_Summary_YYYY-MM-DD.md` (daily progress)
- **Start Here Tomorrow**: `Start_Here_Tomorrow_YYYY-MM-DD.md` (next steps)

### Workflow Templates
- **Location**: `_bmad-output/bmb-creations/synapse/synapse-sidecar/workflows/`
- **Files**: 8 markdown files with detailed guidance and code examples

---

## Research Timeline

**Week 1-2**: Data Understanding + Baselines
- Data profiling, baseline suite, feature engineering

**Week 3-4**: Neural Models + Rule Discovery
- Neural prototypes, calibration, symbolic rule mining

**Week 5-6**: Integration + Optimization
- Hybrid inference, stability policies, K optimization

**Week 7-8**: Evaluation + Reporting
- Ablations, statistical tests, research conclusions

---

## Success Metrics

### Primary
- **Good-or-better rate** @K (target: ≥90%)
- **Unacceptable rate** @K (target: ≤5%)
- **Pool size K*** (target: minimize within [20, 27])

### Secondary
- **Stability**: Jaccard similarity (target: ≥0.70)
- **Calibration**: ECE (target: <0.10)
- **Interpretability**: Rule evidence quality (qualitative assessment)

---

## Team & Working Model

**Project Lead**: y (Research strategy, domain expertise, decision-making)
**Technical Implementation**: AI agents (Synapse + Claude Code)
**Working Model**:
- Agents write all code (Python, algorithms, debugging)
- User executes code with agent guidance
- Agents interpret results and explain technical decisions
- User makes strategic research decisions

**Key Point**: User is NOT a trained programmer/data scientist. Agents handle all technical heavy lifting. User provides research direction and domain context.

---

## Contact / Support

**GitHub Repository**: https://github.com/rogerfiske/c5_Neuro_symbolic.git
**BMAD Documentation**: https://github.com/bmad-code-org/BMAD-METHOD
**Project Lead**: y
**Agent**: Synapse (Neuro-Symbolic ML Research Engineer)

---

## License & Attribution

**Dataset**: CA5 predictive maintenance data (proprietary)
**Framework**: BMAD (Build-Measure-Analyze-Deploy) Method
**Agent Builder**: BMB (BMAD Builder Module)
**PRD**: Initial draft by ChatGPT, enhanced by BMad Master + y

---

## Project Status Summary

| Component | Status | Date Completed |
|-----------|--------|----------------|
| Dataset Acquisition | ✅ Complete | 2026-01-21 |
| PRD Creation | ✅ Complete | 2026-01-22 |
| PRD Enhancement | ✅ Complete | 2026-01-22 |
| Agent Card | ✅ Complete | 2026-01-21 |
| Synapse Agent Build | ✅ Complete | 2026-01-22 |
| Agent Validation | 🔄 In Progress | 2026-01-23 (step 1/6) |
| Agent Installation | ⏳ Pending | 2026-01-24 (planned) |
| Data Profiling | ⏳ Pending | 2026-01-24 (planned) |
| Baseline Suite | ⏳ Pending | 2026-01-24 or later |
| Feature Engineering | ⏳ Pending | Week 1-2 |
| Rule Discovery | ⏳ Pending | Week 3-4 |
| Neural Models | ⏳ Pending | Week 3-4 |
| Hybrid Integration | ⏳ Pending | Week 5-6 |
| K Optimization | ⏳ Pending | Week 5-6 |
| Ablation Study | ⏳ Pending | Week 7-8 |
| Research Conclusion | ⏳ Pending | Week 7-8 |

**Overall Progress**: ~40% infrastructure complete, 0% research complete

---

## Change Log

### 2026-01-23
- Initiated Synapse agent validation workflow
- Completed validation step 1 (v-01-load-review)
- Created validation report tracking document
- Session documentation created for 2026-01-23

### 2026-01-22
- Created Synapse Expert agent (8 workflows, 12 sidecar files)
- Enhanced PRD with code quality and GPU decision logic
- Prepared for validation phase
- Session documentation created

### 2026-01-21
- Project initiated
- Dataset obtained (CA5_date.csv, 11,685 records)
- Initial PRD created by ChatGPT
- Agent card drafted

---

**README Last Updated**: 2026-01-23
**Next Update**: After validation completion and data profiling
