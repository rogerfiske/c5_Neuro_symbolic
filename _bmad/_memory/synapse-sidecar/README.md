# Synapse Sidecar

This folder stores reference materials, workflow templates, and domain knowledge for the **Synapse** Expert agent.

## Purpose

Synapse is a Neuro-Symbolic ML Research Engineer specializing in predictive maintenance for a 5-machine production line. The sidecar contains:
- Reference documentation (PRD, checklists, GPU decision logic)
- 8 research workflow files (data-profile → ablation-report)
- Domain knowledge (dataset schema, tier definitions, artifact standards)
- Quick reference guides (workflow sequence, troubleshooting)

## Structure

```
synapse-sidecar/
├── README.md (this file)
├── instructions.md          # Core execution protocols and operational guidance
├── prd.md                   # Complete PRD for research objectives
├── pre-run-checklist.md     # Template checklist enforced before training workflows
├── gpu-decision-logic.md    # Heuristics for local PC vs RunPod H200
├── dataset-schema.md        # CA5 dataset specification and invariants
├── tier-definitions.md      # Service level definitions (Excellent/Good/Unacceptable)
├── artifact-standards.md    # Output requirements for each workflow
├── workflow-sequence.md     # Visual guide to 8-workflow progression
├── troubleshooting.md       # Common issues and solutions
└── workflows/               # 8 research workflow files
    ├── data-profile.md
    ├── baseline-suite.md
    ├── feature-schema.md
    ├── rulebook-draft.md
    ├── neural-model-prototype.md
    ├── hybrid-inference.md
    ├── k-optimizer.md
    └── ablation-report.md
```

## Runtime Access

After BMAD installation, this folder will be accessible at:
`{project-root}/_bmad/_memory/synapse-sidecar/{filename}.md`

## Workflow Execution Model

Synapse operates through 8 sequential research workflows:
1. **data-profile** → Validate schema, detect gaps, analyze distributions
2. **baseline-suite** → Build strong baselines with tier metrics
3. **feature-schema** → Engineer features with leakage audits
4. **rulebook-draft** → Discover symbolic rules with evidence
5. **neural-model-prototype** → Build calibrated neural scorers
6. **hybrid-inference** → Combine neural + symbolic reasoning
7. **k-optimizer** → Optimize pool size under constraints
8. **ablation-report** → Compare variants systematically

Each workflow produces artifacts: config (YAML/JSON), metrics (CSV/JSON), logs (markdown), visualizations (PNG/HTML).

## Output Location

All workflow artifacts are saved to:
`{project-root}/_bmad-output/synapse/{workflow-name}/{run-id}/`

## Created

- **Date**: 2026-01-22
- **Project**: c5_neuro_symbolic
- **User**: y
