<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Provide ideas on how to utilize 'Neuro-symbolic AI' analyze the uploaded csv dataset.  The dataset may need to be to converted to a format(s) that 'Neuro-symbolic AI' can process.

The dataset has 6 cols:
date,date,P_1,P_2,P_3,P_4,P_5
Where P_1 to P_5 or the 5 part numbers (1-39 inclusive) that are required for a machine running 18 hours/day since installation 9/8/2008. No duplicate part numbers can be shipped on a given day and 5 parts must be shipped.

Perfect! Now I'll generate the comprehensive final report.

# Utilizing Neuro-Symbolic AI for Part Number Dataset Analysis: A Comprehensive Implementation Guide

## Executive Summary

Neuro-symbolic AI represents a transformative paradigm that combines the pattern recognition capabilities of neural networks with the logical reasoning power of symbolic systems. This report provides eight concrete, actionable approaches for applying neuro-symbolic AI to the CA5 part number dataset—a time series containing 3,300+ daily records of machine part selections spanning from September 2008 to present. Each approach transforms the CSV data into formats suitable for neuro-symbolic processing, enabling automated discovery of business rules, anomaly detection, constraint learning, and optimization of part selection processes.[^1_1][^1_2]

The dataset contains five part numbers (P_1 through P_5) selected daily from a range of 1-39, with constraints requiring exactly five unique parts per day. This combinatorial structure, combined with temporal dependencies and business constraints, makes it an ideal candidate for neuro-symbolic approaches that can learn both implicit patterns from data (neural) and explicit logical rules (symbolic).

![Comparison of 8 neuro-symbolic AI approaches for part number dataset analysis across 5 key dimensions. Higher scores are better across all dimensions. Process Mining NeSy4PPM and CP + Neural Forecast show the best overall balance.](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/055e7296d81c12804573bb35fbc10b32/ca72dc5d-20ac-4738-b085-67c4531c4319/5c3699c9.png)

Comparison of 8 neuro-symbolic AI approaches for part number dataset analysis across 5 key dimensions. Higher scores are better across all dimensions. Process Mining NeSy4PPM and CP + Neural Forecast show the best overall balance.

## Understanding Neuro-Symbolic AI

### Foundational Architecture

Neuro-symbolic AI integrates two complementary AI paradigms that address each other's fundamental weaknesses. The **neural component** excels at perception, pattern recognition, and learning from unstructured data through deep learning architectures such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers. However, neural networks suffer from lack of interpretability, data hunger, and poor logical consistency. Conversely, the **symbolic component** leverages formal logic, knowledge representation, and rule-based reasoning to provide structured knowledge, explainability, and systematic inference. Symbolic systems struggle with noisy perceptual data and require explicit knowledge engineering.[^1_1][^1_2][^1_3][^1_4][^1_5]

The integration occurs through three primary architectural patterns:[^1_6]

1. **Sequential (Neural→Symbolic)**: Neural networks process raw data first, then symbolic reasoning operates on the neural outputs. Example: A neural classifier detects parts in an image, then a symbolic reasoner applies business rules to validate the selection.
2. **Parallel (Neural|Symbolic)**: Both components operate independently on the same input, with their outputs combined through fusion mechanisms such as weighted voting or attention-based aggregation[^1_7].
3. **Embedded (Neural[Symbolic])**: Symbolic reasoning is embedded within the neural architecture itself, where logical inference rules become internal to the neural network through differentiable logic operations.[^1_8][^1_9][^1_1]

### Mathematical Framework

At the core lies a sophisticated integration expressed as:[^1_3]

**Hybrid Output = Neural Network Output + Symbolic Reasoning Output**

The neural component processes data through layered transformations with activation functions (ReLU, sigmoid), producing pattern-based predictions. The symbolic component applies logical rules from knowledge bases (represented as first-order logic, temporal logic, or constraint satisfaction problems) to the same problem domain. The synergy emerges from grounding—a mapping from logical symbols to neural representations.[^1_9][^1_10]

In Logic Tensor Networks (LTN), this grounding maps:

- **Constants** → Real-valued tensors (data points)
- **Predicates** → Neural networks outputting truth values in[^1_11]
- **Functions** → Neural networks transforming inputs
- **Logical formulas** → Fuzzy logic computations over neural outputs[^1_12][^1_13][^1_9]


## Data Format Transformations

The CA5_date.csv dataset must be converted into formats compatible with neuro-symbolic processing. Each transformation preserves the core information (date, five part numbers) while restructuring it to support different reasoning paradigms.

### Transformation 1: Tensor Format (for LTN)

**Structure**: Multi-dimensional arrays suitable for differentiable logic[^1_9][^1_13]

```python
# Each row converts to:
date_vector = [day_of_week, day_of_month, days_since_start]  # 3-dim
part_vectors = [one_hot_encoding(P_1), ..., one_hot_encoding(P_5)]  # 5×39 dims
position_encoding = [1, 2, 3, 4, 5]  # ordinal positions
combined_tensor = concatenate([date_vector, flatten(part_vectors), position_encoding])
```

This format enables LTN to ground logical predicates like `selected(part, date)` and `compatible(part_i, part_j)` as neural networks operating on these tensors.[^1_12][^1_14]

### Transformation 2: Event Log Format (for Process Mining)

**Structure**: XES (eXtensible Event Stream) format used by process mining tools[^1_15][^1_16][^1_17]

```xml
<log>
  <trace>
    <string key="concept:name" value="2008-09-08"/>
    <event>
      <string key="concept:name" value="select_part"/>
      <int key="part_number" value="13"/>
      <int key="position" value="1"/>
      <date key="time:timestamp" value="2008-09-08T00:00:00"/>
    </event>
    <!-- Repeat for P_2 through P_5 -->
  </trace>
</log>
```

This transformation treats each date as a "case" and each part selection as an "activity," enabling process discovery algorithms to mine sequential patterns and declarative constraints.[^1_18][^1_17]

### Transformation 3: Knowledge Graph Format

**Structure**: RDF triples representing entities and relationships[^1_4][^1_19]

```turtle
:Part_13 a :Part ;
    :shipped_on :Date_2008_09_08 ;
    :in_position :Position_1 ;
    :frequency 0.23 .

:Part_13 :co_occurs_with :Part_20 ;
    :co_occurrence_weight 0.67 .

:Date_2008_09_08 :precedes :Date_2008_09_09 .
```

Knowledge graphs enable symbolic querying (SPARQL), ontological reasoning (OWL inference), and graph neural network embeddings.[^1_5][^1_20][^1_19]

### Transformation 4: Temporal Graph Format

**Structure**: Dynamic graph with timestamped nodes and edges[^1_21][^1_22]

```python
# For each date:
G_t = {
    'nodes': [13, 20, 23, 29, 30],  # Parts selected on date t
    'edges': [(13,20), (13,23), ..., (29,30)],  # Complete graph
    'node_features': {
        13: [frequency, recency, avg_position],
        20: [frequency, recency, avg_position],
        # ...
    },
    'timestamp': t
}
```

This enables temporal graph neural networks to learn time-aware node embeddings and predict future part selections.[^1_23]

### Transformation 5: Constraint Satisfaction Problem (CSP)

**Structure**: Variables, domains, and constraints[^1_24][^1_25][^1_26]

```python
CSP = {
    'variables': ['P_1', 'P_2', 'P_3', 'P_4', 'P_5'],
    'domains': {
        'P_1': range(1, 40),  # All possible part numbers
        'P_2': range(1, 40),
        # ...
    },
    'constraints': [
        AllDifferent(['P_1', 'P_2', 'P_3', 'P_4', 'P_5']),
        Cardinality(['P_1', ..., 'P_5'], exactly=5),
        # Learned constraints from neural model:
        If('P_1' == 7, then='P_2' in [12, 18, 31])  # Example co-occurrence rule
    ]
}
```

CSP solvers (CP-SAT, OR-Tools) can then find feasible solutions satisfying all constraints.[^1_25]

### Transformation 6: Sequential Trace Format

**Structure**: Time-ordered sequences for sequence-to-sequence models[^1_27][^1_28]

```python
# Sliding window approach:
sequences = [
    [[13,20,23,29,30], [22,25,29,38,39], [4,6,14,26,32], ...],  # 7-day window
    [[22,25,29,38,39], [4,6,14,26,32], [1,4,5,35,37], ...],     # Shifted by 1 day
    # ...
]
```

This format feeds into LSTM/GRU networks for next-day prediction and VAE-based anomaly detection.[^1_29][^1_28]

## Eight Implementation Approaches

### Approach 1: Temporal Constraint Discovery with Logic Tensor Networks

**Neural-Symbolic Integration**: LTN learns neural predicates while maximizing satisfaction of a first-order logic knowledge base.[^1_12][^1_9][^1_13]

**Architecture**:

1. **Neural Predicates** (learnable functions → ):[^1_11]
    - `frequently_paired(part_i, part_j)`: Learned compatibility score
    - `position_preference(part, position)`: Learned positional bias
    - `temporal_follows(part_i, part_j, Δt)`: Temporal correlation within Δt days
2. **Symbolic Knowledge Base** (FOL axioms):

```prolog
∀x,y: selected(x,date) ∧ selected(y,date) ∧ x≠y → compatible(x,y)
∀x: count(selected(x,date)) = 5
∀date,x: selected(x,date) → part_valid(x)  /* x ∈ [1,39] */
```

3. **Training Process**:
    - Ground formulas using historical data (dates as constants)
    - Compute satisfaction degree using fuzzy logic semantics (Łukasiewicz t-norm)[^1_9][^1_10]
    - Loss = 1 - aggregated_satisfaction
    - Backpropagate through differentiable logic operators

**Expected Outputs**:

- **Learned rules**: "IF part_7_selected THEN frequently_paired(7,12) = 0.87" → "Parts 7 and 12 co-occur 87% of the time"
- **Positional patterns**: "position_preference(part_3, P_1) = 0.63" → "Part 3 appears in first position 63% of the time"
- **Temporal dependencies**: "temporal_follows(15, 22, 3) = 0.71" → "Part 15 follows part 22 within 3 days, 71% confidence"

**Implementation**: LTNtorch library (PyTorch-based). Training time: ~15-30 minutes on 3,300 records.[^1_13][^1_14]

**Code Snippet**:

```python
import ltn
import torch

# Define predicates as neural networks
class FrequentlyPaired(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(78, 128),  # 39+39 one-hot inputs
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, part_i, part_j):
        combined = torch.cat([part_i, part_j], dim=-1)
        return self.net(combined)

# Ground predicates
Frequently_Paired = ltn.Predicate(FrequentlyPaired())

# Define axioms
axiom_compatibility = ltn.Forall(
    ltn.diag(parts_i, parts_j),
    ltn.Implies(
        ltn.And(selected(parts_i), selected(parts_j)),
        Frequently_Paired(parts_i, parts_j)
    )
)

# Loss = 1 - satisfaction
loss = 1 - axiom_compatibility.value
```


### Approach 2: Anomaly Detection via Sequential VAE + Symbolic Verification

**Neural-Symbolic Integration**: Variational Autoencoder learns normal sequence patterns; symbolic checker verifies constraints.[^1_28][^1_30]

**Architecture**:

1. **Neural Component**: Sequence-to-Sequence VAE
    - **Encoder**: LSTM maps 7-day part sequences → latent vector z (dimension 64)
    - **Decoder**: LSTM reconstructs sequences from z
    - **Anomaly score**: reconstruction_error + LOF(z) (Local Outlier Factor in latent space)
2. **Symbolic Component**: Constraint Verifier
    - Check decoded sequences for:
        - **AllDifferent violation**: Duplicate parts detected
        - **Cardinality violation**: ≠5 parts
        - **Range violation**: Parts outside[^1_11][^1_31]
3. **Hybrid Anomaly Detection**:

```python
neural_score = ||actual_sequence - reconstructed_sequence||²
symbolic_score = count_constraint_violations(actual_sequence)
final_score = α * neural_score + β * symbolic_score
```


**Workflow**:

1. Train VAE on 80% of data (normal sequences assumed)
2. For each test sequence:
    - Compute reconstruction error
    - Compute latent space outlier score (LOF)
    - Run symbolic constraint checker
    - Flag if final_score > threshold

**Expected Outputs**:

- **Anomaly report**: "Date 2015-03-12: Duplicate part 22 in positions P_2 and P_4 (symbolic violation), reconstruction error = 0.34 (neural anomaly)"
- **Time series visualization**: Plot of anomaly scores over time, highlighting spikes
- **Top-K anomalies**: Ranked list of most anomalous days with explanations

**Implementation**: PyTorch (VAE), scikit-learn (LOF), custom constraint checker.

**Advantage**: Combines unsupervised neural learning (no labeled anomalies needed) with explicit constraint verification (high interpretability).[^1_28]

### Approach 3: Process Mining with Neuro-Symbolic Predictive Monitoring

**Framework**: NeSy4PPM—first Python library integrating neural process prediction with symbolic background knowledge.[^1_17]

**Architecture**:

1. **Data Preprocessing**: Convert CSV → XES event log
2. **Neural Predictor**: LSTM trained on prefix→next_part prediction
    - Input: Sequence of past part selections (e.g., last 5 selections)
    - Output: Probability distribution over 39 possible next parts
3. **Symbolic Knowledge**: DECLARE constraints mined from data[^1_17]
    - **Response**: `response(part_7, part_12)` → "If part 7 selected, part 12 must follow eventually"
    - **Precedence**: `precedence(part_3, part_18)` → "Part 18 can only be selected after part 3"
    - **Co-existence**: `coexist(part_5, part_22)` → "Parts 5 and 22 must both appear or neither"
4. **Constrained Prediction**:
    - LSTM generates candidate next part
    - Symbolic checker verifies DECLARE constraint compliance
    - If violation, reject and sample next-best candidate
    - Ensures predictions satisfy learned business rules

**Expected Outputs**:

- **DECLARE model**: Visual process graph showing part dependencies
- **Violation analysis**: "Historical violations of precedence(3→18): 12 instances (0.4%)"
- **Constrained recommendations**: "Next part should be 12 (LSTM: 0.73 prob, satisfies response(7→12))"

**Implementation**: NeSy4PPM library, PM4Py (process mining), PyTorch.[^1_17]

**Why Recommended**: Best balance of ease (existing library), interpretability (visual DECLARE models), and practical value (discovers actual business rules).

![Suitability scores for each neuro-symbolic approach based on their primary use case. Differentiable ILP excels at rule induction, while VAE leads in anomaly detection, and CP + Neural Forecast is optimal for production optimization.](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/055e7296d81c12804573bb35fbc10b32/10287c33-07aa-4fca-bb99-a7372411aca5/88d85be2.png)

Suitability scores for each neuro-symbolic approach based on their primary use case. Differentiable ILP excels at rule induction, while VAE leads in anomaly detection, and CP + Neural Forecast is optimal for production optimization.

### Approach 4: Temporal Graph Neural Networks for Part Relationship Learning

**Neural-Symbolic Integration**: GNN learns part embeddings from co-occurrence graphs; symbolic rules extracted from embeddings.[^1_21][^1_22][^1_23]

**Architecture**:

1. **Graph Construction**:
    - **Nodes**: 39 parts (static)
    - **Edges**: Created between parts selected on the same day
    - **Edge weights**: Cumulative co-occurrence frequency
    - **Temporal snapshots**: Daily graph states
2. **Temporal GNN**:
    - **Message passing**: Aggregate neighbor information with temporal attention
    - **Node embeddings**: Learn 128-dim vectors capturing part "roles"
    - **Edge prediction**: Predict likely future co-occurrences
3. **Symbolic Rule Extraction**:
    - **Clustering**: K-means on node embeddings → discover part families
    - **Rules**: "Parts in Family_A:  → frequently co-occur"[^1_32][^1_20][^1_33][^1_34]
    - **Constraints**: "At most 3 parts from same family per day"

**Expected Outputs**:

- **Interactive graph visualization**: NetworkX with nodes sized by frequency, edges weighted by co-occurrence
- **Part families**: Discovered clusters (e.g., "High-frequency core parts", "Seasonal parts", "Rare specialty parts")
- **Temporal predictions**: "Given today's selection, part 12 has 0.78 probability tomorrow"[^1_2][^1_35][^1_36][^1_34][^1_32]

**Implementation**: PyTorch Geometric (TGNN), NetworkX (graph ops), visualization with Gephi or Plotly.

### Approach 5: Differentiable Inductive Logic Programming for Rule Induction

**Neural-Symbolic Integration**: Neural networks learn rule weights; symbolic decoder extracts first-order logic programs.[^1_37][^1_38]

**Architecture**:

1. **Rule Templates** (Learnable matrix embeddings W):

```prolog
selected(D, P1, Pos1) ∧ selected(D, P2, Pos2) → paired(P1, P2)
selected(D1, P, _) ∧ precedes(D1, D2) → selected(D2, P, _)  /* Recency */
high_frequency(P) ∧ selected(D, P, P_1) → critical_part(P)
```

2. **Differentiable Semantics**:
    - Convert logic program to neural network via differentiable TP operator (immediate consequence operator)[^1_37]
    - Each rule → differentiable constraint function
    - Loss = inconsistency between induced rules and ground truth data
3. **Symbolic Extraction**:
    - Decode weight matrices W → symbolic rules
    - Prune rules with weight < threshold
    - Output human-readable logic program with confidence scores

**Expected Outputs**:

```prolog
% Learned Rules with Confidence
rule_1(confidence=0.87): selected(D,7,_) ∧ selected(D,12,_) → paired(7,12).
rule_2(confidence=0.73): selected(D1,P,_) ∧ high_frequency(P) ∧ precedes(D1,D2) → selected(D2,P,_).
rule_3(confidence=0.65): selected(D,P,P_1) → position_preference(P,P_1).
```

**Implementation**: Custom (requires implementing differentiable TP operator) or adapt DFOL framework. High complexity but exceptional interpretability.[^1_37]

**Advantage**: Produces pure symbolic rules—the most interpretable neuro-symbolic output possible.[^1_38]

### Approach 6: Signal Temporal Logic for Interpretable Time Series Patterns

**Framework**: Weighted STL (wSTL) + Neural Networks[^1_29]

**Architecture**:

1. **Multi-View Time Series**:
    - **View 1**: Part frequency vectors (39-dim per day): `[count(part_1), ..., count(part_39)]`
    - **View 2**: Position distributions (5-dim per day)
    - **View 3**: Derivative (rate of change in part usage)
2. **Weighted STL Formulas** (each neuron = subformula):
    - **Eventually**: `F[0,7](part_7_selected)` → "Part 7 selected within 7 days"
    - **Always**: `G[0,30](count(part_12) ≤ 3)` → "Part 12 used ≤3 times per month"
    - **Until**: `part_5_selected U part_18_selected` → "Part 5 continuously until part 18 appears"
3. **Neural STL Classifier**:
    - Each neuron outputs quantitative satisfaction degree ∈[^1_11]
    - Hierarchical composition: atomic formulas → complex temporal patterns
    - Differentiable STL semantics enable gradient-based training[^1_29]

**Expected Outputs**:

```
Learned Formula φ:
  G[0,14](part_3_frequency > 0.2) ∧ F[0,7](part_7 ∧ part_12) 
  → normal_pattern

Natural Language Translation:
  "If part 3 is consistently used (>20% frequency over 14 days) 
   AND parts 7 and 12 co-occur within 7 days, 
   THEN the selection pattern is classified as normal."
```

**Implementation**: Custom wSTL semantics (based on ) or TensorFlow STL extensions. High complexity but produces highly interpretable temporal logic formulas.[^1_29]

### Approach 7: Constraint Programming + Neural Demand Forecasting

**Neural-Symbolic Integration**: LSTM forecasts demand; CP-SAT solver generates feasible, optimized selection plans.[^1_25][^1_39][^1_40]

**Architecture**:

1. **Neural Forecaster**:
    - LSTM predicts next 7 days' demand for each of 39 parts
    - Input: 30-day historical part usage (39×30 tensor)
    - Output: Demand probabilities p(part_i | day_t) for t ∈ [T+1, T+7]
2. **Constraint Solver** (Google OR-Tools CP-SAT):[^1_25]

```python
# Variables
P = {P_1, P_2, P_3, P_4, P_5} for each day t ∈ [T+1, T+7]

# Constraints
AllDifferent(P_1, ..., P_5) for each day
Cardinality(P) = 5 for each day

# Objective: Minimize deviation from neural forecast
minimize: Σ_{t,i} |selected(part_i, t) - demand_forecast(part_i, t)|
```

3. **Hybrid Workflow**:
    - Train LSTM on historical data
    - Generate 7-day demand forecast
    - Formulate CSP with forecast as soft objective
    - Solve CSP → feasible selection plan satisfying all constraints
    - Validate: check learned temporal rules (e.g., "part_7 requires part_12 within 2 days")

**Expected Outputs**:

- **7-day optimized schedule**: Feasible part selections maximizing demand coverage
- **Demand vs selection comparison**: Visualization showing forecast adherence
- **Constraint satisfaction report**: "All 35 constraints satisfied (5 per day × 7 days)"
- **Optimization metrics**: "Forecast adherence: 87%, constraint violations: 0"

**Implementation**: PyTorch (LSTM), Google OR-Tools CP-SAT. Medium complexity, high practical value for production planning.[^1_25]

**Why Valuable**: Directly applicable to real-world scheduling—ensures predictions are **always feasible** (no constraint violations).[^1_39][^1_40]

### Approach 8: Knowledge Graph + Graph Neural Network Reasoning

**Neural-Symbolic Integration**: Knowledge graph stores explicit relationships; GNN learns latent embeddings for similarity-based reasoning.[^1_4][^1_20][^1_19]

**Architecture**:

1. **Knowledge Graph Construction** (Neo4j or RDFLib):
    - **Entities**: Part(39), Date(3,300+), Position(5), PartFamily(learned)
    - **Relations**: `:SHIPPED_ON`, `:IN_POSITION`, `:CO_OCCURS_WITH` (weighted), `:PRECEDES`, `:BELONGS_TO_FAMILY`
    - **Ontology**: OWL axioms defining part taxonomies, temporal relations
2. **GNN Encoder**:
    - Multi-relational GNN (R-GCN) learns node embeddings
    - Different message passing functions per relation type
    - Node embeddings capture semantic part similarity
3. **Symbolic Reasoning**:
    - **SPARQL queries**:

```sparql
SELECT ?part WHERE {
  ?part :co_occurs_with :Part_7 .
  ?edge :weight ?w .
  FILTER(?w > 0.5)
}
```

    - **OWL inference**: Infer part categories based on co-occurrence patterns
    - **Rule-based**: `IF part_A :belongs_to_family Family_1 AND part_B :belongs_to_family Family_1 THEN compatible(A,B)`
4. **Neuro-Symbolic Queries**:
    - Combine GNN similarity + symbolic constraints:

```python
# Find parts similar to part_12 (neural) that satisfy constraint X (symbolic)
candidates = gnn.most_similar(part_12, top_k=10)
valid = [p for p in candidates if symbolic_checker.satisfies(p, constraint_X)]
```


**Expected Outputs**:

- **Interactive knowledge graph**: Neo4j Browser visualization with 3,300+ nodes
- **Part similarity matrix**: Cosine similarity from GNN embeddings
- **Inferred relationships**: "Part_7 :is_compatible_with Part_12 (inferred from co-occurrence pattern)"
- **Query results**: "Parts semantically similar to part_3 not used in last 30 days: "[^1_3][^1_33][^1_41]

**Implementation**: Neo4j (graph database), PyTorch Geometric (R-GCN), RDFLib (ontology). Medium complexity, high interpretability through visual graph exploration.

## Implementation Roadmap

### Phase 1: Data Preparation (Week 1)

**Objective**: Transform CA5_date.csv into analysis-ready formats.

**Tasks**:

1. **Data Validation**:

```python
import pandas as pd

df = pd.read_csv('CA5_date.csv')

# Check constraints
assert df[['P_1','P_2','P_3','P_4','P_5']].apply(
    lambda row: len(set(row)) == 5, axis=1
).all(), "Duplicate parts detected"

assert df[['P_1','P_2','P_3','P_4','P_5']].isin(range(1,40)).all().all(), \
    "Parts outside [1,39] range"
```

2. **Feature Engineering**:
    - Temporal: `day_of_week`, `month`, `days_since_start`
    - Part-level: `part_frequency` (rolling 30-day), `recency`, `position_distribution`
    - Sequence: 7-day sliding windows, change point detection
3. **Train/Test Split** (temporal, not random):
    - Training: 2008-2015 (70%)
    - Validation: 2016 (15%)
    - Test: 2017 (15%)

**Deliverable**: Preprocessed datasets in multiple formats (tensors, event log, graph, time series).

### Phase 2: Prototype Development (Week 2-3)

**Objective**: Implement minimal viable version of chosen approach.

**Recommended Starting Point**: **Approach 3 (Process Mining NeSy4PPM)**

**Rationale**:

- Existing Python library (low implementation barrier)[^1_17]
- Visual, interpretable outputs (DECLARE models)
- Practical value (discovers actual business rules)
- Fast training (~5-10 minutes)

**Implementation Steps**:

1. **Install Dependencies**:

```bash
pip install pm4py NeSy4PPM torch
```

2. **Convert to Event Log**:

```python
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

# Reshape: each part selection = separate event
events = []
for idx, row in df.iterrows():
    for i, part in enumerate(row[['P_1','P_2','P_3','P_4','P_5']]):
        events.append({
            'case:concept:name': row['date'],
            'concept:name': f'select_part_{part}',
            'part_number': part,
            'position': i+1,
            'time:timestamp': pd.to_datetime(row['date'])
        })

event_log = log_converter.apply(pd.DataFrame(events))
```

3. **Train Neural Predictor**:

```python
from NeSy4PPM import NeSyPPM

model = NeSyPPM(
    encoder='index-based',
    neural_model='LSTM',
    hidden_dim=128
)

model.train(event_log, epochs=50)
```

4. **Mine DECLARE Constraints**:

```python
from pm4py.algo.discovery.declare import algorithm as declare_miner

declare_model = declare_miner.apply(event_log)
# Output: response(part_7, part_12), precedence(part_3, part_18), ...
```

5. **Generate Constrained Predictions**:

```python
prefix = ['select_part_7', 'select_part_3']
next_part = model.predict_constrained(prefix, declare_model)
# Returns: part_12 (satisfies response(7→12) with 0.87 confidence)
```


**Deliverable**: Working prototype with discovered DECLARE rules and constrained predictions.

### Phase 3: Evaluation \& Iteration (Week 4)

**Objective**: Quantify performance and refine approach.

**Metrics**:

1. **Accuracy**: Next-part prediction accuracy (neural component)
2. **Constraint Satisfaction Rate**: % of predictions satisfying learned DECLARE rules (symbolic component)
3. **Rule Quality**: Expert validation of discovered rules (human-in-the-loop)
4. **Interpretability Score**: Clarity of explanations (user study)

**Evaluation Protocol**:

- Compare neural-only LSTM vs neuro-symbolic NeSy4PPM
- Expected improvement: 10-20% accuracy gain from symbolic constraints[^1_17]
- Measure constraint violation rate (should approach 0% with neuro-symbolic)

**Iteration**:

- If accuracy low: Add more LSTM layers, increase hidden dimensions
- If constraint violations high: Strengthen symbolic checker, add hard constraints
- If interpretability low: Extract simpler DECLARE rules, visualize process graphs


### Phase 4: Scaling \& Deployment (Week 5-6)

**Objective**: Scale to full dataset, integrate into decision support system.

**Tasks**:

1. **Full-Scale Training**: Train on entire 2008-2015 dataset (~2,500 records)
2. **Real-Time Inference**: Deploy model as REST API for daily predictions
3. **Monitoring Dashboard**: Visualize:
    - Predicted vs actual part selections
    - Constraint satisfaction trends over time
    - Anomaly alerts (if VAE component integrated)
4. **Expert Review Loop**: Weekly review sessions with domain experts to validate discovered rules

**Deliverable**: Production-ready neuro-symbolic system with monitoring and explainability.

## Critical Success Factors

### Technical Requirements

1. **Computational Resources**:
    - **Minimal**: CPU-only laptop (for Approach 3, 7)
    - **Recommended**: GPU with 8GB VRAM (for Approach 1, 2, 4, 6)
    - **Training time**: 10 minutes (Approach 3) to 2 hours (Approach 5)
2. **Python Libraries**:

```bash
# Core
pip install torch torchvision pandas numpy matplotlib scikit-learn

# Neuro-Symbolic Frameworks
pip install LTNtorch  # Logic Tensor Networks
pip install pm4py NeSy4PPM  # Process Mining
pip install torch-geometric  # Graph Neural Networks
pip install ortools  # Constraint Programming

# Knowledge Graphs
pip install neo4j rdflib networkx
```

3. **Data Quality**: Validate constraints (no duplicates, exactly 5 parts, range ) before training.[^1_11][^1_31]

### Interpretability Best Practices

Neuro-symbolic AI's primary advantage is explainability. To maximize interpretability:

1. **Always Extract Symbolic Rules**: Don't just train neural networks—decode learned weights into human-readable logic.[^1_37][^1_38]
2. **Visualize Reasoning Paths**: Show how neural predictions + symbolic constraints combine to produce final output.[^1_42][^1_17]
3. **Provide Counterfactual Explanations**: "Prediction changed because part_7 violated precedence(3→7) constraint".[^1_28]
4. **Validate with Domain Experts**: Schedule weekly review sessions to verify discovered rules align with business knowledge.

### Common Pitfalls to Avoid

1. **Temporal Data Leakage**: Always use temporal train/test splits, never random shuffling (would leak future information into training).[^1_43]
2. **Overfitting to Constraints**: If symbolic constraints too strict, model may overfit to training data. Use soft constraints with learned weights.[^1_42]
3. **Ignoring Neural Component**: Purely symbolic systems can't discover new patterns. The neural component is essential for learning from data.[^1_6]
4. **Black-Box Neural Networks**: If neural component not grounded to symbolic logic, lose interpretability. Use LTN or differentiable ILP to maintain explainability.[^1_12][^1_9]

## Comparative Analysis: Choosing the Right Approach

### Decision Framework

| **If Your Priority Is...** | **Choose Approach...** | **Rationale** |
| :-- | :-- | :-- |
| **Discovering hidden business rules** | 5 (Differentiable ILP) | Produces pure symbolic logic rules with confidence scores[^1_37][^1_38] |
| **Detecting anomalous part selections** | 2 (VAE Anomaly Detection) | Neural VAE learns normal patterns; symbolic checker flags constraint violations[^1_28][^1_30] |
| **Optimizing production schedule** | 7 (CP + Neural Forecast) | Ensures all predictions are feasible (satisfy constraints)[^1_25][^1_39] |
| **Understanding process workflows** | 3 (Process Mining NeSy4PPM) | Visual DECLARE models show part dependencies[^1_17] |
| **Learning part relationships** | 8 (Knowledge Graph + GNN) | Captures both explicit (graph) and latent (GNN embeddings) relationships[^1_4][^1_19] |
| **Minimal implementation effort** | 3 (Process Mining NeSy4PPM) | Existing library, 100 lines of code[^1_17] |
| **Maximum interpretability** | 5 or 6 (ILP or STL) | Output is pure symbolic logic or temporal formulas[^1_29][^1_37] |
| **Scalability to millions of records** | 2, 3, or 7 | LSTM/VAE scale well; CP solvers handle large problem sizes[^1_25][^1_39] |

### Hybrid Combinations

For maximum power, combine multiple approaches:

1. **Anomaly Detection + Process Mining**:
    - Use Approach 2 (VAE) to detect anomalies
    - Use Approach 3 (Process Mining) to discover **why** (which DECLARE rules violated)
2. **Forecasting + Optimization**:
    - Use Approach 7 (CP + Neural) for daily scheduling
    - Use Approach 1 (LTN) to learn long-term constraints
3. **Knowledge Graph + GNN + ILP**:
    - Use Approach 8 (KG + GNN) to learn part embeddings
    - Use Approach 5 (ILP) to extract rules from embeddings
    - Store rules back in knowledge graph for symbolic querying

## Advanced Extensions

### Extension 1: Multi-Agent Neuro-Symbolic System

Integrate multiple specialized agents, each with neural perception + symbolic reasoning:[^1_44][^1_7]

- **Agent 1**: Demand forecasting (LSTM + temporal logic constraints)
- **Agent 2**: Inventory optimization (GNN + constraint solver)
- **Agent 3**: Anomaly detection (VAE + rule verifier)
- **Coordination**: Symbolic mediator (fibring) combines agent outputs[^1_6]

**Benefit**: Modular, scalable, each agent specializes in one task.[^1_6]

### Extension 2: Incorporating External Knowledge

Enrich with domain ontologies:[^1_2][^1_45][^1_46]

- **Part Ontology**: Categorize parts (mechanical, electrical, consumable)
- **Process Ontology**: Manufacturing workflows, assembly sequences
- **Temporal Ontology**: Seasonality, maintenance cycles

**Integration**: Add ontological axioms to LTN knowledge base or knowledge graph, enabling richer symbolic reasoning.[^1_4][^1_20]

### Extension 3: Causal Discovery

Extend temporal GNN with causal inference:[^1_2]

- Learn causal graphs: "Part 7 selection **causes** part 12 selection (not just correlation)"
- Use interventional reasoning: "What if we stop using part 3? Impact on part 18 usage?"
- Counterfactual analysis: "If we had selected part 22 instead of 31 on date X, would anomaly have occurred?"

**Method**: Integrate causal graphs into symbolic knowledge base, train GNN to respect causal structure.[^1_2]

## Conclusion and Strategic Recommendations

Neuro-symbolic AI offers a transformative approach to analyzing the CA5 part number dataset by combining the pattern recognition power of neural networks with the logical reasoning capabilities of symbolic systems. The eight implementation approaches presented span a spectrum from process mining (practical, easy to implement) to differentiable ILP (cutting-edge, highly interpretable) to constraint optimization (production-ready scheduling).

**Primary Recommendation**: Begin with **Approach 3 (Process Mining with NeSy4PPM)** for its optimal balance of ease, interpretability, and practical value. This enables rapid prototyping (1-2 weeks) with immediate actionable insights (discovered DECLARE rules showing part dependencies).[^1_17]

**Secondary Recommendation**: For production optimization, implement **Approach 7 (CP + Neural Forecasting)** to generate feasible, constraint-satisfying part selection schedules that maximize demand coverage while respecting all learned business rules.[^1_25][^1_39][^1_40]

**Advanced Recommendation**: For research-grade rule discovery, invest in **Approach 5 (Differentiable ILP)** to extract human-readable first-order logic rules with confidence scores—the pinnacle of interpretable AI.[^1_37][^1_38]

The fundamental insight of neuro-symbolic AI is that neither neural networks nor symbolic systems alone suffice for complex real-world problems. Neural networks excel at discovering latent patterns but lack transparency. Symbolic systems provide explicit reasoning but require hand-crafted knowledge. By integrating both paradigms, we achieve AI systems that learn from data (neural), reason with logic (symbolic), and explain their decisions (neuro-symbolic)—precisely what's needed to extract actionable intelligence from the CA5 part number dataset.[^1_1][^1_2][^1_6]

The dataset's combinatorial structure (5 parts from 39, no duplicates), temporal dependencies (daily sequences spanning years), and implicit business constraints make it an ideal testbed for neuro-symbolic approaches. Whether the goal is anomaly detection, process discovery, demand forecasting, or constraint learning, the methods outlined in this report provide concrete, implementable pathways to transform raw CSV data into interpretable, actionable knowledge.

This report synthesizes findings from 104 sources spanning neuro-symbolic AI architectures, temporal logic frameworks, constraint programming, process mining, knowledge graphs, and graph neural networks. Key foundational sources include:

- **Neuro-Symbolic Foundations**:  on hybrid architectures[^1_1][^1_2][^1_3][^1_5]
- **Logic Tensor Networks**:  on differentiable logic[^1_12][^1_9][^1_13][^1_14]
- **Temporal Logic**:  on LTL and STL for sequences[^1_27][^1_33][^1_29]
- **Process Mining**:  on event log analysis[^1_15][^1_16][^1_18][^1_17]
- **Constraint Programming**:  on CSP and optimization[^1_24][^1_25][^1_26][^1_39]
- **Graph Neural Networks**:  on temporal graph learning[^1_21][^1_22][^1_23]
- **Differentiable ILP**:  on neural rule induction[^1_37][^1_38]
- **Knowledge Graphs**:  on symbolic knowledge representation[^1_4][^1_20][^1_19]
