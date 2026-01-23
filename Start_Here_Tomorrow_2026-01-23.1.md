# Start Here Tomorrow - January 24, 2026

## Quick Context Refresh

**Yesterday's Achievement**: Initiated Synapse agent validation workflow (completed step 1 of 6)
**Today's Mission**: Complete validation, fix any issues, install agent, begin data profiling
**Current Position**: Validation step v-01-load-review.md complete, awaiting [C] selection

---

## Where You Left Off

You initiated the validation workflow and are at this menu:

```
Is this the correct agent to validate and is it identified as the proper type?

[A] Advanced Elicitation
[P] Party Mode
[C] Yes, Begin Validation
```

**Decision**: You chose to select **[C] Yes, Begin Validation** today (deferred from yesterday).

---

## Today's Plan (Recommended Sequence)

### Phase 1: Complete Validation (30-60 minutes)

**Starting Point**: Select [C] to continue validation workflow

**Validation Steps That Will Execute**:
1. ✅ v-01-load-review.md (COMPLETED yesterday)
2. ⏳ v-02a-validate-metadata.md - Check metadata properties (id, name, title, icon, module, hasSidecar)
3. ⏳ v-02b-validate-persona.md - Verify persona field purity (role/identity/communication_style/principles)
4. ⏳ v-03-validate-menu.md - Validate menu structure and command triggers
5. ⏳ v-04-validate-structure.md - Check YAML structure and syntax
6. ⏳ v-05-validate-sidecar.md - Verify sidecar files exist and are complete
7. ⏳ v-06-report.md - Generate final validation report with recommendations

**Expected Outcomes**:
- Comprehensive validation report at `_bmad-output/bmb-creations/validation-report-synapse.md`
- List of any errors, warnings, or issues
- Clear guidance on fixes (if needed)
- Confirmation of agent readiness for installation

---

### Phase 2: Address Any Issues (15-45 minutes, if needed)

**If validation finds issues**:
1. Review validation report findings
2. Apply recommended fixes to agent YAML or sidecar files
3. Re-run validation to confirm fixes
4. Proceed when validation passes cleanly

**If validation passes cleanly**:
- Proceed directly to Phase 3

---

### Phase 3: Agent Installation (15-30 minutes)

**Goal**: Make Synapse available in your BMAD project

**Recommended Approach**: Full installation via custom module

**Steps**:
1. Create module structure:
   ```
   C:\Users\Minis\CascadeProjects\c5_neuro_symbolic\_bmad\custom-agents\
   ├── module.yaml (unitary: true)
   └── agents\
       └── synapse\
           ├── synapse.agent.yaml
           └── synapse-sidecar\
               ├── README.md
               ├── instructions.md
               ├── prd.md
               ├── pre-run-checklist.md
               └── workflows\ (8 files)
   ```

2. Create module.yaml:
   ```yaml
   module:
     id: custom-agents
     name: Custom Research Agents
     description: Project-specific agents for C5 research
     unitary: true
   ```

3. Copy agent files from build location to module location

4. Test activation: `/synapse`

**Expected Outcome**: Synapse activates successfully with all sidecar files loaded

---

### Phase 4: Git Initial Commit (10-15 minutes)

**Goal**: Push project to GitHub for version control

**Note**: Path discrepancy to resolve first - build output shows "c5_neuro_symmetric" but project is "c5_neuro_symbolic"

**Steps**:
1. Verify correct project directory
2. Review .gitignore (already created)
3. Run git commands:
   ```bash
   cd C:\Users\Minis\CascadeProjects\c5_neuro_symbolic
   git init
   git add .
   git commit -m "Initial commit: Synapse agent built and validated, ready for research"
   git remote add origin https://github.com/rogerfiske/c5_Neuro_symbolic.git
   git branch -M main
   git push -u origin main
   ```

**Expected Outcome**: Project backed up to GitHub

---

### Phase 5: First Workflow - Data Profiling (2-4 hours)

**Goal**: Thoroughly understand CA5 dataset before modeling

**Prerequisites**:
- Synapse installed and activatable
- Python dependencies installed: `pip install pandas matplotlib scipy numpy scikit-learn`

**Command**: Activate Synapse with `/synapse`, then type `DP` or `data-profile`

**How This Works** (Remember: You're NOT writing code):
1. Synapse will write the complete Python script for data profiling
2. You'll review the script with Synapse's explanation
3. You'll execute the script (copy-paste to terminal or run .py file)
4. Synapse will interpret results and generate the report
5. You'll review findings and approve next steps

**What This Workflow Analyzes**:
- Schema validation (5 parts per day, IDs 1-39, no duplicates)
- Calendar gaps (weekends, COVID period, other irregularities)
- Part frequency distributions (near-uniform?)
- Temporal patterns and trends
- Concept drift detection
- Data quality issues

**Artifacts Produced**:
```
_bmad-output/synapse/data-profile/{run-id}/
├── config.yaml           # Run parameters, seeds, timestamp
├── metrics.csv           # Quantitative findings
├── run_log.md           # Execution trace
└── visualizations/
    ├── part_frequency.png
    ├── temporal_trends.png
    └── gap_analysis.png
```

**Expected Outcome**: Comprehensive understanding of dataset characteristics to inform baseline development

---

## Critical Files to Review

### 1. Validation Report (After Phase 1)
**Location**: `_bmad-output/bmb-creations/validation-report-synapse.md`
**Why**: Understand any issues found and fixes needed

### 2. Agent YAML (If Fixes Needed)
**Location**: `_bmad-output/bmb-creations/synapse/synapse.agent.yaml`
**Why**: Apply any corrections identified by validation

### 3. Sidecar Files (For Reference)
**Location**: `_bmad-output/bmb-creations/synapse/synapse-sidecar/`
**Why**: Verify all workflow files and reference docs are complete

### 4. Pre-Run Checklist (Before Data Profiling)
**Location**: `synapse-sidecar/pre-run-checklist.md`
**Why**: Familiarize with research rigor requirements

---

## Known Issues / Watch-Outs

### 1. Path Discrepancy
**Issue**: Build output location shows "c5_neuro_symmetric" but project is "c5_neuro_symbolic"
**Impact**: May need to verify correct paths during installation
**Action**: Check if this was intentional or a typo; use correct "symbolic" path for installation

### 2. Validation May Find Issues
**Issue**: First validation run may identify problems
**Impact**: Will need to fix before installation
**Action**: Review validation report carefully and apply all recommended fixes

### 3. Python Dependencies
**Issue**: Data profiling requires specific libraries
**Impact**: Script will fail if dependencies missing
**Action**: Install before running first workflow: `pip install pandas matplotlib scipy numpy scikit-learn`

---

## Decision Points You May Encounter

### Decision 1: How to Handle Validation Failures
**Question**: If validation finds errors, fix immediately or defer?
**Recommendation**: Fix immediately while validation context is fresh
**Reasoning**: Issues compound if deferred; clean validation enables smooth installation

### Decision 2: Installation Method
**Question**: Custom module vs project-local vs BMAD global?
**Recommendation**: Custom module in project `_bmad/custom-agents/`
**Reasoning**: Keeps agent with project, allows version control, enables easy updates

### Decision 3: Data Profiling Depth
**Question**: Quick analysis or comprehensive deep dive?
**Recommendation**: Comprehensive deep dive (use full workflow template)
**Reasoning**: Foundation for all subsequent research; shortcuts here create downstream issues

### Decision 4: Local PC or RunPod for Profiling
**Question**: Where to run data profiling?
**Recommendation**: Local PC
**Reasoning**: Data profiling is CPU-bound pandas/matplotlib work; no GPU needed; saves H200 cost

---

## Success Criteria for Today

### Minimum Success (Must Achieve)
✅ Validation completed with all steps executed
✅ Validation report generated
✅ Any critical errors identified and documented
✅ Clear path forward established (fixes or installation)

### Target Success (Aim For)
✅ Minimum success criteria met
✅ All validation issues resolved
✅ Synapse agent installed and activatable
✅ Git initial commit pushed to GitHub
✅ Data profiling workflow initiated

### Stretch Success (If Extra Time)
✅ Target success criteria met
✅ Data profiling completed with full report
✅ Dataset characteristics documented
✅ Baseline suite workflow initiated

---

## Quick Command Reference

**During Validation**:
- Select `C` to begin validation steps
- Review each step's findings
- Select `C` to continue to next step

**After Installation**:
```
/synapse          - Activate Synapse agent
MH or help        - Show menu with all commands
DP or data-profile - Start data profiling workflow
CH or chat        - Chat with Synapse about anything
DA or exit        - Dismiss Synapse agent
```

**Git Commands**:
```bash
git init
git add .
git commit -m "Initial commit: Synapse agent built and validated"
git remote add origin https://github.com/rogerfiske/c5_Neuro_symbolic.git
git branch -M main
git push -u origin main
```

---

## Troubleshooting Quick Reference

### Issue: Validation fails with path errors
**Symptoms**: Can't find sidecar files
**Checks**:
1. Are sidecar files in `_bmad-output/bmb-creations/synapse/synapse-sidecar/`?
2. Do paths in agent YAML use correct `{project-root}` format?
**Solution**: Verify sidecar location matches YAML references

### Issue: Agent won't activate after installation
**Symptoms**: `/synapse` command not recognized
**Checks**:
1. Was installation completed correctly?
2. Is module.yaml in correct location?
3. Did you restart BMAD or reload configuration?
**Solution**: Review installation steps, verify file structure

### Issue: Python dependencies missing
**Symptoms**: ImportError during data profiling
**Checks**:
1. Are pandas, matplotlib, scipy installed?
2. Is correct Python environment activated?
**Solution**: `pip install pandas matplotlib scipy numpy scikit-learn`

---

## Motivational Notes

### Why Today Matters
- **Validation** ensures your agent is production-ready and error-free
- **Installation** makes Synapse your active research partner
- **Data profiling** is the foundation for ALL modeling decisions
- **Git commit** protects your work and enables collaboration

### What You're Building Toward
- Research-grade validation ensures reliable workflows
- Proper installation enables seamless agent activation
- Dataset understanding prevents costly modeling mistakes
- Version control protects weeks of research investment

### Remember
You're the research strategist. AI agents (Synapse, Claude Code) handle all technical implementation. Your role is to guide direction, approve approaches, and make strategic decisions based on agent analysis.

**You've built the infrastructure. Today you validate it and begin the science.** 🧪

---

## End-of-Day Checklist for Today

Before stopping tonight, ensure:
- [ ] Validation completed with comprehensive report generated
- [ ] Any validation issues addressed and resolved
- [ ] Synapse agent installed and activatable (if validation passed)
- [ ] Git initial commit pushed to GitHub
- [ ] Data profiling workflow initiated or completed
- [ ] Key findings documented (if profiling completed)
- [ ] Session summary created for today
- [ ] Start-here doc updated for next session

---

## Resources at Your Disposal

### Documentation
- Validation report: `_bmad-output/bmb-creations/validation-report-synapse.md` (after validation)
- Session summary (yesterday): `Session_Summary_2026-01-22.md`
- Project memory: `.project_memory.md`
- README: `README.md`

### Agent Files
- Agent YAML: `_bmad-output/bmb-creations/synapse/synapse.agent.yaml`
- Sidecar folder: `_bmad-output/bmb-creations/synapse/synapse-sidecar/` (12 files)

### Data
- Dataset: `data/raw/CA5_date.csv` (11,685 records)

### Tools
- Local PC: AMD Ryzen 9 6900HX, 64GB RAM, AMD Radeon RX 6600M 8GB
- RunPod H200: 141GB HBM3 (reserve for large models)

---

## Final Reminders

1. **Validation First**: Don't skip validation - it catches issues early
2. **Fix Issues Immediately**: Don't defer validation fixes
3. **Trust the Process**: Step-by-step validation is thorough for good reason
4. **Install Properly**: Use custom module approach for clean integration
5. **Document Findings**: Data profiling insights inform all future work

---

**Good luck, y!**

**Today's Mission**: Validate → Fix (if needed) → Install → Commit → Profile
**Tomorrow's Mission**: Baseline Suite → Feature Engineering

**You're ~40% done with infrastructure, ready to start research today.**

---

**Document Created**: 2026-01-23 (end of day)
**For Session**: 2026-01-24 (tomorrow)
**Status**: Ready to resume with [C] selection to begin validation
