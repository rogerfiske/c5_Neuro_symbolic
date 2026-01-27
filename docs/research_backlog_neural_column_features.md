# Research Backlog: Neural Column-Position Features

**Created**: 2026-01-27
**Status**: Planned
**Priority**: High - Could improve model accuracy

---

## Background

Analysis revealed that each column (m_1 to m_5) has distinct part distributions:

| Column | 95% Range | Top Part | Characteristic |
|--------|-----------|----------|----------------|
| m_1 | 1-18 (17 parts) | Part 1 (12.8%) | Low part IDs |
| m_2 | 2-25 (24 parts) | Part 10 (6.1%) | Low-mid range |
| m_3 | 7-33 (27 parts) | Part 19 (4.9%) | Middle range |
| m_4 | 15-38 (24 parts) | Part 30 (5.8%) | Mid-high range |
| m_5 | 22-39 (18 parts) | Part 39 (13.3%) | High part IDs |

However, **per-column frequency baseline did NOT outperform global baseline** (68.3% vs 68.7% GoB). This suggests the insight needs to be incorporated into the neural model rather than used for frequency-based prediction.

---

## Research Direction 1: Position Embeddings per Column

### Concept
Add learnable position embeddings that encode which column (m_1-m_5) a part appeared in, allowing the model to learn column-specific patterns.

### Implementation Approach
```python
class ColumnPositionEmbedding(nn.Module):
    def __init__(self, num_columns=5, embed_dim=128):
        super().__init__()
        self.column_embedding = nn.Embedding(num_columns, embed_dim)

    def forward(self, part_embeddings, column_indices):
        # part_embeddings: (batch, seq_len, 5, embed_dim)
        # column_indices: (5,) = [0, 1, 2, 3, 4]
        col_emb = self.column_embedding(column_indices)  # (5, embed_dim)
        return part_embeddings + col_emb.unsqueeze(0).unsqueeze(0)
```

### Expected Benefit
- Model can learn that Part 1 in m_1 has different significance than Part 1 in m_5
- Captures column-specific base rates implicitly

### Complexity
Low - Minor modification to existing architecture

---

## Research Direction 2: Separate Attention Heads per Column

### Concept
Use multi-head attention where each head specializes in one column's patterns, then aggregate.

### Implementation Approach
```python
class ColumnSpecificAttention(nn.Module):
    def __init__(self, embed_dim=128, num_columns=5):
        super().__init__()
        # One attention head per column
        self.column_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads=1)
            for _ in range(num_columns)
        ])
        self.fusion = nn.Linear(embed_dim * num_columns, embed_dim)

    def forward(self, x):
        # x: (seq_len, batch, 5, embed_dim)
        outputs = []
        for col_idx, head in enumerate(self.column_heads):
            col_x = x[:, :, col_idx, :]  # (seq_len, batch, embed_dim)
            attn_out, _ = head(col_x, col_x, col_x)
            outputs.append(attn_out)

        combined = torch.cat(outputs, dim=-1)
        return self.fusion(combined)
```

### Expected Benefit
- Each head can specialize in patterns relevant to its column
- May capture column-specific temporal dynamics

### Complexity
Medium - Requires architecture restructuring

---

## Research Direction 3: Column-Position Features

### Concept
Add explicit features indicating column position to the input representation.

### Implementation Approach
```python
def add_column_features(sequence):
    """
    sequence: (batch, seq_len, 5) - part IDs
    returns: (batch, seq_len, 5, num_features)
    """
    batch, seq_len, num_cols = sequence.shape

    features = []
    for col_idx in range(num_cols):
        # One-hot column position
        col_onehot = torch.zeros(num_cols)
        col_onehot[col_idx] = 1

        # Column-specific statistics (precomputed)
        col_mean_part = COLUMN_MEANS[col_idx]  # e.g., [6.7, 13.2, 19.9, 26.6, 33.2]
        col_std_part = COLUMN_STDS[col_idx]

        features.append(torch.cat([col_onehot, col_mean_part, col_std_part]))

    return features
```

### Expected Benefit
- Provides explicit signal about column context
- Allows model to weight predictions by column reliability

### Complexity
Low - Feature engineering only

---

## Research Direction 4: Ensemble (Global + Per-Column)

### Concept
Combine global neural predictions with per-column predictions using learned weights.

### Implementation Approach
```python
class EnsemblePredictor(nn.Module):
    def __init__(self, global_model, column_models):
        super().__init__()
        self.global_model = global_model
        self.column_models = nn.ModuleList(column_models)  # 5 models
        self.ensemble_weights = nn.Parameter(torch.ones(6) / 6)  # learnable

    def forward(self, x):
        global_logits = self.global_model(x)  # (batch, 39)

        # Per-column predictions (each predicts for its valid parts)
        column_logits = []
        for col_idx, model in enumerate(self.column_models):
            col_logits = model(x[:, :, col_idx])  # (batch, num_valid_parts[col])
            # Map to full 39-part space
            full_logits = map_to_full_space(col_logits, VALID_PARTS[col_idx])
            column_logits.append(full_logits)

        # Weighted combination
        weights = F.softmax(self.ensemble_weights, dim=0)
        combined = weights[0] * global_logits
        for i, col_logits in enumerate(column_logits):
            combined += weights[i+1] * col_logits

        return combined
```

### Expected Benefit
- Leverages both global patterns and column-specific expertise
- Learnable weights can adapt to what works best

### Complexity
High - Requires training multiple models and ensemble layer

---

## Research Direction 5: Per-Column Output Heads

### Concept
Single encoder, but 5 separate output heads that predict parts likely for each column.

### Implementation Approach
```python
class MultiHeadPredictor(nn.Module):
    def __init__(self, encoder, embed_dim=128, num_parts=39):
        super().__init__()
        self.encoder = encoder

        # Separate prediction head per column
        self.column_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_parts)
            for _ in range(5)
        ])

        # Aggregation layer
        self.aggregator = nn.Linear(num_parts * 5, num_parts)

    def forward(self, x):
        context = self.encoder(x)  # (batch, embed_dim)

        # Get predictions from each column head
        col_preds = [head(context) for head in self.column_heads]

        # Option A: Average
        combined = torch.stack(col_preds).mean(dim=0)

        # Option B: Concatenate and learn aggregation
        # combined = self.aggregator(torch.cat(col_preds, dim=-1))

        return combined
```

### Expected Benefit
- Each head can specialize without separate training
- Shared encoder captures common patterns

### Complexity
Medium - Straightforward extension of current architecture

---

## Recommended Priority Order

1. **Column-Position Features** (Direction 3) - Lowest effort, can test quickly - **IMPLEMENTED**
2. **Position Embeddings** (Direction 1) - Low effort, principled approach - **IMPLEMENTED**
3. **Per-Column Output Heads** (Direction 5) - Medium effort, good potential - **IMPLEMENTED**
4. **Ensemble** (Direction 4) - Higher effort, may not beat simpler approaches - **IMPLEMENTED** (via fusion gate)
5. **Separate Attention Heads** (Direction 2) - Highest effort, uncertain benefit - DEFERRED

## Implementation Status (2026-01-27)

All approaches implemented in `runpod_package/models/column_enhanced.py`:

| Approach | Class/Config | Status |
|----------|--------------|--------|
| Column Position Embeddings | `ColumnAwareEmbedding` | Ready |
| Column-Position Features | `ColumnFeatureEmbedding` | Ready |
| Per-Column Output Heads | `PerColumnOutputHeads` | Ready |
| Ensemble (via fusion gate) | Built into output heads | Ready |
| Separate Attention | `ColumnSeparateAttention` | Implemented but not tested |

Training script: `runpod_package/train_column_enhanced.py`
Local test: `scripts/test_column_enhanced_model.py` - All 6 configs PASS

---

## Success Criteria

| Metric | Current Best | Target |
|--------|--------------|--------|
| GoB @K=30 | 69.9% (Hybrid) | >71% |
| Part 12 Recall | 36.5% (baseline) | >50% |
| Excellent Rate | 27.0% | >28% |

---

## Prerequisites

- RunPod GPU access for training experiments
- Baseline model checkpoint (`outputs/best_model/`)
- Modified training pipeline to support new architectures

---

## Next Steps

1. Start with Direction 3 (Column-Position Features) - can be tested locally
2. If promising, move to Direction 1 (Position Embeddings) on RunPod
3. Document findings and iterate

---

**Document Last Updated**: 2026-01-27
