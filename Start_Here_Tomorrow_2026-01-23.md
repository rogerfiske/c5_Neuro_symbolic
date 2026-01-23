# Start Here Tomorrow - January 23, 2026

## Quick Context Refresh

**Yesterday's Achievement**: Completed creation of "Synapse" - Expert BMAD agent for neuro-symbolic AI research
**Today's Mission**: Validate Synapse agent and begin first research workflow (Data Profiling)

**Git Repository Created**: https://github.com/rogerfiske/c5_Neuro_symbolic.git (initial push today)

**IMPORTANT**: You're the research project lead guiding strategy and decisions. The AI agents (Synapse, Claude, etc.) will write all code, implement algorithms, and handle technical details. You'll execute scripts with agent guidance and make strategic decisions based on results.

---

## Where You Left Off

You were at the **celebration step** of the Agent Builder workflow with this menu:

```
[V] Run Validation - Comprehensive quality checks on agent YAML
[S] Skip - Complete Now - Finish workflow, agent is ready!
[A] Advanced Elicitation - Deep dive for additional insights
[P] Party Mode - Celebrate with multiple agent perspectives
```

**Decision**: You chose to stop for the day and resume with **[V] Run Validation** tomorrow.

---

## Today's Plan (Recommended Sequence)

### Phase 0: Git Initial Commit (10-15 minutes)
**Goal**: Push project to GitHub for version control

**Steps**:
1. Review what files to commit (I'll help you create .gitignore)
2. Run git commands:
   ```bash
   cd C:\Users\Minis\CascadeProjects\c5_neuro_symbolic
   git init
   git add .
   git commit -m "Initial commit: Synapse agent created, ready for validation"
   git remote add origin https://github.com/rogerfiske/c5_Neuro_symbolic.git
   git branch -M main
   git push -u origin main
   ```

**Expected Outcome**: Project backed up to GitHub

---

### Phase 1: Validation (30-60 minutes)
**Goal**: Ensure Synapse agent YAML is production-ready

**Steps**:
1. Resume the Agent Builder workflow where you left off
2. Select **[V] Run Validation** option
3. Review validation report for any issues
4. Fix any syntax errors or structural problems identified
5. Re-validate if changes were made

**Expected Outcome**: Clean validation report with no errors

---

### Phase 2: Agent Installation (15-30 minutes)
**Goal**: Make Synapse available in your BMAD project

**Steps**:
1. Review agent files at:
   ```
   C:\Users\Minis\CascadeProjects\c5_neuro_symmetric\_bmad-output\bmb-creations\synapse\
   ├── synapse.agent.yaml
   └── synapse-sidecar/ (12 files)
   ```

2. **Option A: Quick Test (Development)**
   - Copy agent files to project-local location for testing
   - Test activation without full installation

3. **Option B: Full Installation (Recommended)**
   - Create custom module structure:
     ```
     my-research-agents/
     ├── module.yaml (unitary: true)
     └── agents/
         └── synapse/
             ├── synapse.agent.yaml
             └── synapse-sidecar/
     ```
   - Install via BMAD installer or "Modify BMAD Installation"

**Expected Outcome**: Synapse accessible via `/synapse` command

---

### Phase 3: First Workflow - Data Profiling (2-4 hours)
**Goal**: Thoroughly understand CA5 dataset before modeling

**Command**: Once Synapse is activated, type: `DP` or `data-profile`

**How This Works**:
- Synapse will **write the complete Python script** for data profiling
- You'll **review the script** with Synapse's explanation
- You'll **execute the script** (copy-paste to terminal or run .py file)
- Synapse will **interpret results** and generate the report
- You'll **review findings** and approve next steps

**What This Workflow Does**:
1. Loads CA5 dataset (11,685 records)
2. Validates invariants (5 unique parts/day, IDs 1-39)
3. Detects calendar gaps (weekends, COVID period)
4. Analyzes part frequency distributions
5. Identifies temporal patterns and concept drift
6. Generates comprehensive data quality report

**Artifacts Produced**:
```
_bmad-output/synapse/data-profile/{run-id}/
├── config.yaml
├── metrics.csv
├── data_profile_report.md
├── part_frequency.png
└── temporal_trends.png
```

**Key Questions to Answer**:
- Are all invariants satisfied?
- How many calendar gaps exist?
- Is part frequency distribution near-uniform?
- Is there evidence of concept drift?
- What temporal patterns are visible?

**Expected Outcome**: Comprehensive understanding of dataset characteristics

---

### Phase 4: Baseline Suite (Optional - if time permits, 2-4 hours)
**Goal**: Establish performance floor before neural models

**Command**: `BL` or `baseline-suite`

**What This Workflow Does**:
1. Implements 5 baseline models:
   - Frequency baseline (global part counts)
   - Recency baseline (time since last use)
   - Last-N-days baseline (frequency in recent window)
   - Co-occurrence baseline (part associations)
   - Weighted recency-frequency
2. Evaluates all baselines with tier metrics (Excellent/Good/Unacceptable) for K ∈ [20, 27]
3. Identifies best baseline as benchmark to beat

**Expected Outcome**: Clear performance targets for neural/symbolic models

---

## Critical Files to Review Before Starting

### 1. Synapse Agent YAML
**Location**: `_bmad-output/bmb-creations/synapse/synapse.agent.yaml`
**Why**: Understand agent structure, commands, critical actions

### 2. Pre-Run Checklist
**Location**: `_bmad-output/bmb-creations/synapse/synapse-sidecar/pre-run-checklist.md`
**Why**: Familiarize with research rigor requirements before first workflow

### 3. Data Profile Workflow
**Location**: `_bmad-output/bmb-creations/synapse/synapse-sidecar/workflows/data-profile.md`
**Why**: Review workflow structure and expected outputs

### 4. PRD (for context)
**Location**: `docs/prd_neurosymbolic_ai_ca5_v1_1.md`
**Why**: Refresh on project objectives and evaluation metrics

---

## Quick Command Reference

Once Synapse is activated:

```
MH or help        - Show menu with all commands
CH or chat        - Chat with Synapse about anything
DP or data-profile - Start data profiling workflow
DA or exit        - Dismiss Synapse agent
```

---

## Known Issues / Watch-Outs

### 1. Sidecar Path Resolution
**Issue**: YAML uses `{project-root}/_bmad/_memory/synapse-sidecar/` paths
**Watch**: Ensure paths resolve correctly during activation
**Fix**: If path errors occur, verify BMAD installation and sidecar location

### 2. Workflow File Complexity
**Issue**: Workflow templates are detailed but may need adaptation
**Watch**: Python code examples assume certain libraries (pandas, matplotlib, scipy)
**Fix**: Install dependencies: `pip install pandas matplotlib scipy numpy scikit-learn`

### 3. GPU Decision Logic
**Issue**: Workflows recommend RunPod H200 for large models
**Watch**: Local PC adequate for baselines and small prototypes
**Action**: Review GPU decision logic in each workflow before committing to H200

### 4. Output Folder Creation
**Issue**: Workflows assume output folders exist
**Watch**: First workflow may need to create `_bmad-output/synapse/` directory structure
**Fix**: Workflows should include folder creation logic (already in templates)

---

## Success Criteria for Today

### Minimum Success (Must Achieve)
✅ Synapse agent validates without errors
✅ Agent activates successfully in BMAD
✅ Data profiling workflow executes completely
✅ Data quality report generated with all artifacts

### Target Success (Aim For)
✅ Minimum success criteria met
✅ Dataset fully characterized (gaps, drift, distributions)
✅ Baseline suite workflow started or completed
✅ Performance benchmarks established for neural models

### Stretch Success (If Extra Time)
✅ Target success criteria met
✅ Feature schema workflow initiated
✅ Leakage audit strategy defined
✅ Feature engineering plan documented

---

## Decision Points You May Encounter

### Decision 1: Installation Method
**Question**: Quick test or full installation?
**Recommendation**: Full installation (via custom module) for production-grade workflow
**Reasoning**: Properly installed agents integrate cleanly with BMAD infrastructure

### Decision 2: Local PC or RunPod H200 for Data Profiling
**Question**: Where to run first workflow?
**Recommendation**: Local PC (data profiling is CPU-bound EDA)
**Reasoning**: No GPU needed for pandas/matplotlib operations; save H200 for neural training

### Decision 3: Baseline Depth
**Question**: How many baselines to implement?
**Recommendation**: Start with 3 (frequency, recency, co-occurrence), add others if time permits
**Reasoning**: 3 baselines sufficient to establish performance floor; can expand later

### Decision 4: Workflow Customization
**Question**: Use templates as-is or adapt?
**Recommendation**: Start with templates, adapt as you learn dataset characteristics
**Reasoning**: Templates provide structure; customization comes naturally during execution

---

## Resources at Your Disposal

### Documentation
- Session summary (this file's companion): `Session_Summary_2026-01-22.md`
- Agent plan: `_bmad-output/bmb-creations/agent-plan-synapse.md`
- Handoff doc: `docs/handoff_synapse_agent_to_builder.md`
- Enhanced PRD: `docs/prd_neurosymbolic_ai_ca5_v1_1.md`

### Agent Files
- Agent YAML: `_bmad-output/bmb-creations/synapse/synapse.agent.yaml`
- Sidecar folder: `_bmad-output/bmb-creations/synapse/synapse-sidecar/` (12 files)

### Data
- Dataset: `data/raw/CA5_date.csv` (11,685 records)
- Date range: 1992-02-04 to 2026-01-21
- Format: date, m_1, m_2, m_3, m_4, m_5 (5 unique parts per day)

### Tools
- **Local PC**: AMD Ryzen 9 6900HX, 64GB RAM, AMD Radeon RX 6600M 8GB
- **RunPod H200**: 141GB HBM3 (reserve for large models)
- **BMAD Framework**: Agent orchestration and workflow management

---

## Troubleshooting Quick Reference

### Issue: Agent won't activate
**Symptoms**: Error when typing `/synapse`
**Checks**:
1. Was validation successful?
2. Is agent installed in correct location?
3. Are sidecar paths resolving?
**Solution**: Review validation report, check installation steps

### Issue: Workflow file not found
**Symptoms**: Error loading workflow (e.g., data-profile.md)
**Checks**:
1. Are sidecar files in correct location?
2. Do paths use `{project-root}/_bmad/_memory/synapse-sidecar/workflows/` format?
**Solution**: Verify sidecar folder location and path variables

### Issue: Python dependencies missing
**Symptoms**: ImportError during workflow execution
**Checks**:
1. Are pandas, matplotlib, scipy installed?
2. Is Python environment activated?
**Solution**: `pip install pandas matplotlib scipy numpy scikit-learn`

### Issue: Output folder creation fails
**Symptoms**: Can't write to `_bmad-output/synapse/`
**Checks**:
1. Does `_bmad-output` directory exist?
2. Do you have write permissions?
**Solution**: Create directory manually or check permissions

---

## Motivational Notes

### Why Today Matters
- **Validation** ensures your hard work yesterday is production-ready
- **Data profiling** is the foundation for ALL subsequent research
- **Baselines** establish the performance bar to beat
- **Reproducibility** starts on day one (not retrofitted later)

### What You're Building Toward
- Research-grade neuro-symbolic AI system
- Publication-quality ablation studies
- Production-viable predictive maintenance solution
- Reusable research framework for future ML projects

### Remember Synapse's Principles
1. "Start with simple, strong baselines; beat them with evidence"
2. "Reproducibility is non-negotiable"
3. "Interpretability is an engineering requirement"

**You've built the infrastructure. Now it's time to do the science.** 🚀

---

## End-of-Day Checklist for Today

Before stopping tonight, ensure:
- [ ] Synapse agent validated successfully
- [ ] Agent installed and activatable
- [ ] Data profiling completed with report generated
- [ ] Artifacts organized in `_bmad-output/synapse/data-profile/{run-id}/`
- [ ] Key findings documented (gaps, drift, distributions)
- [ ] Next workflow identified (likely: baseline-suite)
- [ ] Session summary created for today
- [ ] Start-here doc updated for next session

---

## Contact / Support

**BMAD Documentation**: https://github.com/bmad-code-org/BMAD-METHOD
**Agent Builder Guide**: `_bmad/bmb/workflows/agent/` (local files)
**Validation Workflow**: `_bmad/bmb/workflows/agent/steps-v/` (will be accessed today)

---

## Final Reminders

1. **Coffee First**: Data profiling is detail-oriented work. Be caffeinated.
2. **Trust the Checklist**: Pre-run checklist prevents 90% of ML research failures.
3. **Document as You Go**: Don't wait until end of day to write findings.
4. **Celebrate Small Wins**: First dataset profiled = first research milestone.
5. **Ask Synapse for Help**: The agent is there to guide you. Use `CH` to chat.

---

**Good luck, y! You've got this.** 💪

**Today's Mission**: Validate → Install → Profile → (Baseline)
**Tomorrow's Mission**: (Baseline) → Feature Engineering → Rule Discovery

**You're 40% done with agent creation, 0% done with research. Let's change that today.**

---

**Document Created**: 2026-01-22 (end of day)
**For Session**: 2026-01-23 (tomorrow)
**Status**: Ready to resume with [V] Validation
