# PRD: Neuro-symbolic AI for Predictive Maintenance Scheduling + Next‑Day Part Pooling

## 1) Overview

### 1.1 Problem
You have historical daily part-ship records for 5 identical machines. Each day includes **exactly 5 unique parts** drawn from **1–39** (no duplicates within a day). The operational need is to stage an **optimal pool of parts** near machines so next-day demand is covered with high confidence while reducing inventory burden.

### 1.2 Objectives
1. **Next-day part pool prediction**: Output a ranked pool of parts (target size **20–27**) likely to include all parts needed tomorrow.
2. **Pool-size optimization**: Choose the pool size that balances:
   - service level / coverage (miss rate)
   - inventory and handling cost
   - optional expedite / downtime costs
3. **Predictive maintenance scheduling**: Infer latent wear/maintenance states and recommend proactive actions, using interpretable “why” explanations.
4. **Interpretability via neuro-symbolic AI**: Discover and apply human-readable rules/constraints alongside statistical learning.

### 1.3 Dataset Snapshot (from the uploaded CSV)
- Records: **11,685 rows**
- Date range: **1992-02-04** to **2026-01-21**
- Columns: `date, m_1, m_2, m_3, m_4, m_5`
- Part domain: **1..39**
- Invariants verified:
  - Exactly 5 parts per day
  - No duplicate parts within a day

---

## 2) Users, Use Cases, and Outputs

### 2.1 Primary Users
- Maintenance planners
- Parts / logistics planners
- Reliability engineers (RCA / pattern discovery)

### 2.2 Core Use Cases
1. **Daily staging plan**
   - Input: history up to latest day D
   - Output:
     - `pool_parts`: list of parts (size K, default K in [20..27])
     - `ranked_parts`: full ranking 1..39 with calibrated scores
     - `rationale`: rules + signals that influenced pool membership

2. **Maintenance scheduling**
   - Output:
     - `maintenance_risk_score` (per day)
     - `recommended_actions` (e.g., inspect subsystem, replace component group)
     - `evidence`: learned rules/events and time-since-last signals

### 2.3 Non-goals (initial phase)
- Real-time streaming ingestion
- Integration into ERP/WMS systems (can be added later)
- Per-machine unique predictions (initially, day-level pooled demand only)

---

## 3) Neuro-symbolic Approach

### 3.1 Why Neuro-symbolic Here
- **Neural**: learns subtle temporal signals, non-linear interactions, and long-range dependencies in a near-uniform part frequency environment.
- **Symbolic**: enforces hard constraints, captures interpretable temporal rules, supports debugging and trust (especially for maintenance actions).

### 3.2 Key Concept: Two-Tier Decisioning
**Tier A — Neural scoring layer**
- Produces per-part probability / score for next-day inclusion and other latent signals.

**Tier B — Symbolic reasoning + optimization layer**
- Enforces invariants (and discovered rules), applies abductive explanations, and selects the pool size K by minimizing expected cost while meeting service-level targets.

---

## 4) Data Representation + Conversion for Neuro-symbolic Learning

### 4.1 Canonical Representations
1. **Multi-hot vector per day (39-dim)**
   - `x_t[p]=1` if part p shipped on day t, else 0
   - Works well for temporal neural models (Transformer/GRU) and differentiable logic constraints.

2. **Event-log / relational facts (symbolic)**
   - `used(date=t, part=p).`
   - `gap_days(date=t, delta=d).` (to account for missing calendar days)
   - Derived facts:
     - `tslu(t,p)=days_since_last_use`
     - `burst(t,p)=consecutive_days_used`
     - `cooccur(p,q)` counts and confidence
     - `follow_within(p,q,Δ)` temporal association (p tends to be followed by q within Δ days)

3. **Knowledge graph**
   - Nodes: `Part`, `Day`, optional `LatentState`
   - Edges: `used_on`, `follows`, `cooccurs`, `time_since_last`
   - Embeddings from GNNs can feed the neural scorer; symbolic rules can also query the graph.

### 4.2 Derived Features (important for rules + maintenance)
- **Recency**: time since last use for each part
- **Seasonality**: day-of-week, month, holiday proxy (optional)
- **Change-point flags**: shifts in usage statistics or recency distributions
- **Association graph metrics**: part centrality, community membership
- **Entropy / dispersion**: how “surprising” today’s set was vs recent window
- **Repetition motif indicators**: n-gram motifs in the daily part sets
- **Calendar gap handling**: explicit representation of skipped days

---

## 5) Symbolic Layer: Rules, Constraints, and Reasoning

### 5.1 Hard Constraints (always enforced)
- Next-day demand is exactly **5 unique** parts.
- Pool is a subset of `{1..39}`.
- Pool size `K` is constrained (default `[20..27]`, configurable).

### 5.2 Soft Constraints / Preferences (learned or configured)
- **Co-occurrence constraints**
  - If `p` appears, `q` tends to appear within N days ⇒ increase `q` score.
- **Mutual inhibition** (if discovered)
  - `p` and `q` rarely appear close together ⇒ reduce joint selection.
- **Motif rules**
  - “After pattern A occurs, pattern B becomes more likely within Δ days.”
- **Maintenance state rules**
  - “High usage of group G followed by long silence may indicate preventive replacement cycle.”

### 5.3 Rule Learning Candidates
1. **Inductive Logic Programming (ILP)** (e.g., Aleph / Popper)
   - Learns clauses like:
     - `next_used(p) :- used_today(q), follow_within(q,p,7), recent(q).`
2. **Probabilistic logic** (e.g., Markov Logic Networks / ProbLog)
   - Captures uncertainty in rules with weights.
3. **Differentiable logic** (e.g., Logic Tensor Networks / Neural Logic Machines)
   - Learns soft logic constraints end-to-end with the neural scorer.

### 5.4 Abductive Explanations for Maintenance
- Introduce a small set of latent “wear modes” or “maintenance regimes”
  - `state_t ∈ {normal, wear_mode_A, wear_mode_B, post_service, ...}`
- Abductive reasoning
  - Find the most plausible state transitions that explain observed sequences.
- Maintenance scheduling
  - Trigger inspection / service if inferred state risk increases.

---

## 6) Neural Layer: Temporal Scoring Models

### 6.1 Baseline Models (fast iteration)
- Logistic regression with recency + window features (per-part)
- Gradient boosting on handcrafted features
- Simple temporal models (GRU/LSTM)

### 6.2 Neuro-symbolic-Compatible Models (recommended)
1. **Transformer on multi-hot sequence**
   - Input: last W days as multi-hot + recency features
   - Output: per-part logits (39)
2. **Temporal Point Process / Neural Hawkes**
   - Treat part occurrences as events; learn excitation/inhibition between parts
3. **Graph + temporal hybrid**
   - GNN embeddings on part association graph + temporal model over days

### 6.3 Calibration (critical for pool sizing)
- Calibrate per-part probabilities
  - Platt scaling / isotonic regression / temperature scaling
- Goal: probabilities reflect true coverage likelihood, enabling stable K optimization.

---

## 7) Pool Selection and Pool-Size Optimization

### 7.1 Prediction Output
- `p_hat[p]`: calibrated probability that part p appears tomorrow
- `ranked_parts`: parts sorted by `p_hat`

### 7.2 Pool Construction
- Default: take top-K parts.
- Neuro-symbolic enhancement
  - Adjust scores using symbolic rules (boost/suppress).
  - Ensure rule-consistent selection (e.g., include follower parts if rule confidence is high).

### 7.3 Choosing the Optimal K
Define a cost objective aligned to **tiered service levels**:
- `C_stock(K)`: inventory + handling cost increasing with K
- `C_tier(m)`: penalty based on # of missing parts `m = 5 - hits`:
  - `m = 0` (5/5) → **Excellent** → penalty ≈ 0
  - `m = 1` (4/5) → **Good** → small/medium penalty (e.g., expedited single part)
  - `m >= 2` (≤3/5) → **Unacceptable** → very large penalty (avoid)
- `C_thrash`: optional penalty for daily pool churn (stability)
- `P_tier(K)`: estimated probabilities for tiers under pool size K

Optimization options (research stage):
1. **Constrained optimization**
   - Choose smallest `K` such that:
     - `P(Excellent | K) >= S_excellent_target` (optional)
     - `P(Good-or-better | K) >= S_good_target` (primary)
     - `P(Unacceptable | K) <= U_max` (hard constraint)
   - Add stability constraint: Jaccard(pool_t, pool_{t-1}) >= J_min (except for strong-confidence shifts)
2. **Expected-cost minimization**
   - Choose `K* = argmin_K [ C_stock(K) + E[C_tier(m) | K] + C_thrash ]`

Practical plan:
- Start with K in **[20, 27]** as the initial band.
- Use rolling-origin backtesting to estimate tier rates and stability.
- Select the smallest K meeting **Good-or-better** and **Unacceptable** constraints.

### 7.4 “Reverse Prediction” (least-likely parts) as a complement
- Predict exclusion set `E` (size 12–19) least likely tomorrow.
- Pool is `P = {1..39} \ E`.
- Symbolic rules naturally express exclusions (“if condition, exclude p”).

---

## 8) Predictive Maintenance Scheduling

### 8.1 Maintenance Signals (from this dataset only)
With only daily part shipments (no sensor telemetry), scheduling is inferred from patterns such as:
- rising recency pressure (parts reappearing sooner)
- regime shifts (change-points in sequence motifs)
- “bursty” part clusters indicating subsystem stress
- post-service signatures (a reset motif after heavy usage)

### 8.2 Outputs
- `risk_score_t`: likelihood that maintenance is due within H days
- `mode_t`: inferred wear mode / regime label
- `recommended_window`: suggested service window
- `explanations`:
  - top rules fired
  - key recency + change-point drivers
  - similar historical episodes (“case-based” retrieval)

### 8.3 Candidate Methods
- Neuro-symbolic HMM:
  - HMM/HSMM latent states + symbolic constraints on transitions and observations.
- Neural regime classifier:
  - Transformer produces regime embedding; symbolic rules map embeddings to interpretable labels.

---

## 9) Evaluation Plan

### 9.1 Pool Prediction Metrics
- **Excellent rate@K**: % days where all 5 true parts ⊆ top-K pool (5/5)
- **Good-or-better rate@K**: % days where ≥4 of 5 true parts are in top-K pool
- **Unacceptable rate@K**: % days where ≤3 of 5 true parts are in top-K pool
- **Expected missing parts**: mean # of true parts not in pool
- **Calibration**: reliability curves / ECE per part and overall
- **Stability**: day-to-day Jaccard similarity of pools (avoid thrash)
- **Pool churn**: count of parts added/removed day-to-day
- **Cost proxy**: `K` (smaller is better) + penalty-weighted unacceptable rate

### 9.2 Maintenance Metrics (proxy-based)
- Regime consistency: inferred modes stable + interpretable
- Predictive utility: risk_score correlates with near-future unusual demand patterns
- Explanation fidelity: rule evidence matches model behavior

### 9.3 Backtesting Protocol
- Rolling-origin evaluation (time-series split)
- Multiple horizons:
  - next-day (primary)
  - 2–7 days (maintenance planning support)

---

## 10) System Architecture

### 10.1 Components
1. **Ingestion + Validation**
   - parse CSV, validate invariants (5 unique parts/day)
2. **Feature Builder**
   - multi-hot vectors
   - recency matrices
   - association graph updates
   - symbolic fact base generation
3. **Model Trainer**
   - neural scorer
   - rule learner (ILP / probabilistic logic / differentiable logic)
   - calibrator
4. **Inference (batch)**
   - compute next-day scores
   - apply symbolic reasoning + constraints
   - choose K* and emit pool + explanation
5. **Reporting**
   - daily report outputs (CSV/JSON/Markdown)

### 10.2 Artifacts Produced
- `daily_pool_prediction.csv` (date, K*, pool_parts, coverage_estimate)
- `daily_ranked_parts.csv` (date, part, score, rank)
- `rules.md` (learned rules, confidence, examples)
- `maintenance_recommendations.csv` (date, risk_score, mode, action, rationale)

### 10.3 Code Quality & Runtime Safety Requirements
**ALL scripts and training routines must include:**

1. **Early Stopping Mechanisms**
   - Configurable maximum iterations/epochs with clear termination criteria
   - Validation-based early stopping (patience parameter) for neural training
   - Convergence thresholds for iterative algorithms (rule mining, optimization)

2. **Progress Monitoring**
   - Real-time terminal output for any process expected to run >10 minutes
   - Progress bars showing: current iteration, ETA, key metrics (loss, accuracy, tier rates)
   - Periodic checkpointing (every N iterations) with auto-resume capability

3. **Infinite Loop Detection**
   - Hard iteration counters with maximum bounds (fail-safe limits)
   - Timeout guards using wall-clock time (e.g., max 6 hours for training)
   - Watchdog timers for hanging operations (data loading, external calls)
   - Sanity checks: detect non-improving metrics over M consecutive iterations

4. **Exception Handling & Logging**
   - Informative error messages with context (iteration number, input state)
   - Graceful degradation: save partial results before failing
   - Stack traces logged to file for debugging

5. **Reproducibility Safeguards**
   - Explicit random seed setting (Python, NumPy, PyTorch, etc.)
   - Config snapshots saved with every run (YAML/JSON with timestamp)
   - Run IDs (UUIDs or timestamps) for artifact tracking
   - Git commit hash logging (if in version control)

### 10.4 Computational Resource Strategy (GPU Decision Logic)

**Local PC Specifications:**
- CPU: AMD Ryzen 9 6900HX
- RAM: 64GB @ 2393MHz
- GPU: AMD Radeon RX 6600M (8GB VRAM)
- Storage: 1TB NVMe SSD

**RunPod H200 Availability:**
- GPU: NVIDIA H200 (141GB HBM3)
- Use for compute-intensive workloads when cost-justified

**Resource Allocation Guidelines:**

| Task Category | Recommended Resource | Rationale |
|---------------|---------------------|-----------|
| **Feature Engineering & EDA** | Local PC | CPU-bound; no GPU benefit |
| **Baseline Models** | Local PC | Simple models (frequency, recency, logistic regression, gradient boosting) |
| **Small Neural Prototypes** | Local PC | Models <5M parameters; fits in 8GB VRAM |
| **Full Transformer Training** | **RunPod H200** | 11K+ sequence length; large batch sizes; multi-head attention benefits from HBM3 bandwidth |
| **Knowledge Graph Embedding** | Local PC (if <10K nodes)<br>**H200** (if >10K nodes/edges) | GNN training scales with graph size |
| **ILP / Rule Mining** | Local PC (small search)<br>**H200** (combinatorial explosion) | Depends on search space size |
| **Hyperparameter Sweeps** | **RunPod H200** (if >50 configs) | Parallel runs; time-critical |
| **Ablation Studies** | **RunPod H200** (if >20 variants) | Multiple model comparisons |
| **Inference / Backtesting** | Local PC | Batch inference is fast; 11K predictions in minutes |

**GPU Decision Heuristics (for agents/scripts):**
1. **Check model parameter count**: >5M params → recommend H200
2. **Check training time estimate**: >2 hours on local → recommend H200
3. **Check batch size requirements**: Need >64 batch size for stability → H200 (more VRAM)
4. **Check parallelization opportunity**: >50 independent runs → H200 (cost-effective at scale)
5. **Check memory footprint**: Intermediate activations >6GB → H200 (avoid OOM errors)

**Cost-Efficiency Note:**
- RunPod H200 should be reserved for tasks that benefit from:
  - High memory bandwidth (large transformers, graph models)
  - Large VRAM (141GB allows massive batch sizes)
  - Parallel execution (sweep 100+ configs overnight)
- Avoid H200 for: data preprocessing, plotting, report generation, small model tuning

---

## 11) Risks, Assumptions, and Mitigations

### 11.1 Risks
- Near-uniform global frequencies can make naive models look random.
- Concept drift over decades (process changes, parts redesigns).
- No direct maintenance labels limits supervised scheduling.

### 11.2 Mitigations
- Emphasize recency, sequence motifs, excitation/inhibition dynamics.
- Drift detection + periodic retraining.
- Self-supervised objectives:
  - masked-part prediction
  - next-set reconstruction
  - contrastive motif embeddings
- Conformal set prediction to control coverage.

---

## 12) Phased Delivery Plan (Suggested)

### Phase 0 — Baselines + Data Products
- Multi-hot representation, recency features, rolling backtest harness
- Baseline pool prediction (top-K) + calibration
- Initial K optimization using cost curve

### Phase 1 — Rule Discovery + Neuro-symbolic Pooling
- Mine association + temporal rules
- Add symbolic score adjustments + constraint checks
- Publish explanations: rules fired + similar historical cases

### Phase 2 — Maintenance Regimes
- Add latent regime modeling (HMM/Transformer embedding)
- Abductive explanations + recommended service windows

### Phase 3 — Robustness + Operations
- Drift monitoring
- Automated retraining schedule
- Confidence reports and QA checks

---

## 13) Open Questions (to finalize PRD)
- **Cost tradeoff**: Difficult to quantify; prefer fewer staged parts. Initial research target range K=20–27 and optimize during research.
- **Acceptance tiers** (pool hit-rate):
  Not a hard binary failure; use tiered acceptance:
  - 5/5 in pool = Excellent
  - 4/5 in pool = Good
  - 3/5 or fewer = Unacceptable
- **External maintenance logs**: No additional maintenance events/logs available to join.
- **Calendar effects**: Early years may include some random weekend gaps; COVID period may include gaps/irregularities.
- **Pool stability**: Yes—prefer stability (minimize daily change), but allow strong-confidence shifts.
