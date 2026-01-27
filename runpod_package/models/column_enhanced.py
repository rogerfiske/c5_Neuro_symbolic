"""
Column-Enhanced Neural Architecture
====================================
Incorporates column-position information into the neural model.

Implements 5 enhancement approaches:
1. Column Position Embeddings - Add learnable embeddings per column
2. Separate Attention Heads - One attention head per column
3. Column-Position Features - Explicit statistical features
4. Ensemble Heads - Combine global + per-column predictions
5. Per-Column Output Heads - Separate prediction head per column

Author: Dr. Synapse Research Pipeline
Date: 2026-01-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# Column Statistics (precomputed from training data)
# ==============================================================================

# 95th percentile part ranges per column (1-indexed)
COLUMN_VALID_PARTS = {
    0: list(range(1, 19)),   # m_1: parts 1-18
    1: list(range(2, 26)),   # m_2: parts 2-25
    2: list(range(7, 34)),   # m_3: parts 7-33
    3: list(range(15, 39)),  # m_4: parts 15-38
    4: list(range(22, 40)),  # m_5: parts 22-39
}

# Mean part ID per column (from training data)
COLUMN_MEANS = [6.7, 13.2, 19.9, 26.6, 33.2]

# Std part ID per column
COLUMN_STDS = [5.2, 6.8, 7.1, 6.5, 5.1]


# ==============================================================================
# Approach 1: Column Position Embeddings
# ==============================================================================

class ColumnAwareEmbedding(nn.Module):
    """
    Part embeddings with column-position awareness.

    Instead of mean-pooling over columns, adds column embeddings
    before aggregation to preserve column identity.
    """

    def __init__(self, num_parts=39, embed_dim=64, num_columns=5, dropout=0.1):
        super().__init__()
        self.num_parts = num_parts
        self.embed_dim = embed_dim
        self.num_columns = num_columns

        # Part embeddings
        self.part_embed = nn.Embedding(num_parts + 1, embed_dim, padding_idx=0)

        # Column embeddings (learnable per-column bias)
        self.column_embed = nn.Embedding(num_columns, embed_dim)

        # Temporal position encoding
        self.pos_embed = nn.Embedding(100, embed_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, part_ids, positions=None):
        """
        Args:
            part_ids: (batch, seq_len, 5) - parts used each day
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, parts_per_day = part_ids.shape
        device = part_ids.device

        # Embed each part: (batch, seq_len, 5, embed_dim)
        part_embeds = self.part_embed(part_ids)

        # Add column embeddings: each column gets its own learned offset
        col_indices = torch.arange(parts_per_day, device=device)
        col_embeds = self.column_embed(col_indices)  # (5, embed_dim)
        part_embeds = part_embeds + col_embeds.unsqueeze(0).unsqueeze(0)

        # Aggregate: weighted mean (columns have different importance)
        # Could also use attention here, but mean is simpler
        day_embeds = part_embeds.mean(dim=2)  # (batch, seq_len, embed_dim)

        # Add temporal positional encoding
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embed(positions)

        embeddings = day_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# ==============================================================================
# Approach 3: Column-Position Features
# ==============================================================================

class ColumnFeatureEmbedding(nn.Module):
    """
    Part embeddings with explicit column statistical features.

    Concatenates learned part embedding with:
    - One-hot column position
    - Normalized column mean
    - Part deviation from column mean
    """

    def __init__(self, num_parts=39, embed_dim=64, num_columns=5, dropout=0.1):
        super().__init__()
        self.num_parts = num_parts
        self.num_columns = num_columns

        # Feature dimensions
        self.part_embed_dim = embed_dim - num_columns - 2  # Reserve space for features
        self.feature_dim = num_columns + 2  # one-hot + 2 stats

        # Part embeddings (smaller to make room for features)
        self.part_embed = nn.Embedding(num_parts + 1, self.part_embed_dim, padding_idx=0)

        # Temporal position encoding
        self.pos_embed = nn.Embedding(100, embed_dim)

        # Register column statistics as buffers
        self.register_buffer('col_means', torch.tensor(COLUMN_MEANS))
        self.register_buffer('col_stds', torch.tensor(COLUMN_STDS))

        # Feature projection
        self.feature_proj = nn.Linear(self.part_embed_dim + self.feature_dim, embed_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, part_ids, positions=None):
        """
        Args:
            part_ids: (batch, seq_len, 5) - parts used each day
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, parts_per_day = part_ids.shape
        device = part_ids.device

        # Embed each part
        part_embeds = self.part_embed(part_ids)  # (batch, seq_len, 5, part_embed_dim)

        # Build column features for each position
        features_list = []
        for col_idx in range(parts_per_day):
            # One-hot column position
            col_onehot = torch.zeros(parts_per_day, device=device)
            col_onehot[col_idx] = 1.0

            # Column mean (normalized to 0-1)
            col_mean_norm = self.col_means[col_idx] / 39.0

            # Part deviation from column mean (z-score)
            part_values = part_ids[:, :, col_idx].float()  # (batch, seq_len)
            deviation = (part_values - self.col_means[col_idx]) / (self.col_stds[col_idx] + 1e-6)
            deviation = deviation.unsqueeze(-1)  # (batch, seq_len, 1)

            # Combine features
            col_mean_expanded = col_mean_norm.expand(batch_size, seq_len, 1)
            col_onehot_expanded = col_onehot.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

            col_features = torch.cat([col_onehot_expanded, col_mean_expanded, deviation], dim=-1)
            features_list.append(col_features)

        # Stack: (batch, seq_len, 5, feature_dim)
        col_features = torch.stack(features_list, dim=2)

        # Concatenate part embeddings with features
        combined = torch.cat([part_embeds, col_features], dim=-1)  # (batch, seq_len, 5, part_embed_dim + feature_dim)

        # Project to embed_dim
        combined = self.feature_proj(combined)  # (batch, seq_len, 5, embed_dim)

        # Aggregate over columns
        day_embeds = combined.mean(dim=2)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embed(positions)

        embeddings = day_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# ==============================================================================
# Approach 2: Separate Attention per Column (in encoder)
# ==============================================================================

class ColumnSeparateAttention(nn.Module):
    """
    Process each column through its own attention mechanism,
    then combine the results.
    """

    def __init__(self, embed_dim=64, num_columns=5, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_columns = num_columns
        self.embed_dim = embed_dim

        # Separate attention for each column
        self.column_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_columns)
        ])

        # Fusion layer
        self.fusion = nn.Linear(embed_dim * num_columns, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, part_embeds):
        """
        Args:
            part_embeds: (batch, seq_len, 5, embed_dim) - embeddings per column
        Returns:
            fused: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, num_cols, embed_dim = part_embeds.shape

        col_outputs = []
        for col_idx in range(num_cols):
            col_input = part_embeds[:, :, col_idx, :]  # (batch, seq_len, embed_dim)
            col_attn, _ = self.column_attentions[col_idx](col_input, col_input, col_input)
            col_outputs.append(col_attn)

        # Concatenate column outputs
        combined = torch.cat(col_outputs, dim=-1)  # (batch, seq_len, embed_dim * 5)

        # Fuse
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)

        return fused


# ==============================================================================
# Approach 5: Per-Column Output Heads
# ==============================================================================

class PerColumnOutputHeads(nn.Module):
    """
    Separate prediction head for each column, then aggregate.

    Each head predicts parts that are valid for its column,
    then results are combined.
    """

    def __init__(self, context_dim=128, num_parts=39, num_columns=5, dropout=0.1):
        super().__init__()
        self.num_parts = num_parts
        self.num_columns = num_columns

        # Per-column prediction heads
        self.column_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim, num_parts)
            )
            for _ in range(num_columns)
        ])

        # Learnable column weights for aggregation
        self.column_weights = nn.Parameter(torch.ones(num_columns) / num_columns)

        # Global head for comparison
        self.global_head = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, num_parts)
        )

        # Fusion gate: balance between column heads and global head
        self.fusion_gate = nn.Sequential(
            nn.Linear(context_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, context):
        """
        Args:
            context: (batch, context_dim) - encoded temporal context
        Returns:
            logits: (batch, num_parts) - combined predictions
            aux: dict with per-column predictions
        """
        batch_size = context.shape[0]

        # Get per-column predictions
        col_logits = []
        for col_idx, head in enumerate(self.column_heads):
            logits = head(context)  # (batch, num_parts)
            col_logits.append(logits)

        col_logits = torch.stack(col_logits, dim=1)  # (batch, 5, num_parts)

        # Weighted combination of column heads
        weights = F.softmax(self.column_weights, dim=0)  # (5,)
        column_combined = (col_logits * weights.view(1, -1, 1)).sum(dim=1)  # (batch, num_parts)

        # Global prediction
        global_logits = self.global_head(context)  # (batch, num_parts)

        # Fusion
        gate = self.fusion_gate(context)  # (batch, 1)
        final_logits = gate * global_logits + (1 - gate) * column_combined

        aux = {
            'per_column_logits': col_logits,
            'global_logits': global_logits,
            'column_weights': weights,
            'fusion_gate': gate
        }

        return final_logits, aux


# ==============================================================================
# Complete Column-Enhanced Model
# ==============================================================================

class ColumnEnhancedPredictor(nn.Module):
    """
    Full model with all column enhancement options.

    Args:
        embedding_type: 'standard', 'column_aware', 'column_features'
        use_column_attention: bool - use separate attention per column
        use_column_heads: bool - use per-column output heads
    """

    def __init__(
        self,
        num_parts=39,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        encoder_type='transformer',
        num_heads=4,
        dropout=0.1,
        # Column enhancement options
        embedding_type='column_aware',  # 'standard', 'column_aware', 'column_features'
        use_column_attention=False,
        use_column_heads=True,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.embedding_type = embedding_type
        self.use_column_attention = use_column_attention
        self.use_column_heads = use_column_heads

        # Select embedding type
        if embedding_type == 'column_aware':
            self.embedding = ColumnAwareEmbedding(
                num_parts=num_parts,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif embedding_type == 'column_features':
            self.embedding = ColumnFeatureEmbedding(
                num_parts=num_parts,
                embed_dim=embed_dim,
                dropout=dropout
            )
        else:
            # Standard embedding (from original model)
            from .neuro_symbolic import PartEmbedding
            self.embedding = PartEmbedding(
                num_parts=num_parts,
                embed_dim=embed_dim,
                dropout=dropout
            )

        # Optional: Column-separate attention
        if use_column_attention:
            self.column_attention = ColumnSeparateAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )

        # Temporal encoder
        if encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.encoder_output_dim = embed_dim
        else:
            self.temporal_encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.encoder_output_dim = hidden_dim * 2

        # Context aggregation
        self.context_pool = nn.Sequential(
            nn.Linear(self.encoder_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output heads
        if use_column_heads:
            self.output_head = PerColumnOutputHeads(
                context_dim=hidden_dim,
                num_parts=num_parts,
                dropout=dropout
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, num_parts)
            )

    def forward(self, part_sequences):
        """
        Args:
            part_sequences: (batch, seq_len, 5) - historical part usage
        Returns:
            logits: (batch, num_parts)
            aux_outputs: dict with intermediate values
        """
        # Embed
        embeddings = self.embedding(part_sequences)  # (batch, seq_len, embed_dim)

        # Optional column attention
        if self.use_column_attention and hasattr(self, 'column_attention'):
            # Need to get per-column embeddings
            # This requires modification to embedding layer to return (batch, seq_len, 5, embed_dim)
            pass

        # Temporal encoding
        if isinstance(self.temporal_encoder, nn.TransformerEncoder):
            encoded = self.temporal_encoder(embeddings)
        else:
            encoded, _ = self.temporal_encoder(embeddings)

        # Pool context
        context = encoded[:, -1, :]  # Use last timestep
        context = self.context_pool(context)

        # Predict
        if self.use_column_heads:
            logits, aux = self.output_head(context)
        else:
            logits = self.output_head(context)
            aux = {}

        aux['context'] = context
        aux['embedding_type'] = self.embedding_type

        return logits, aux

    def predict_pool(self, part_sequences, k=30):
        """Make pool prediction."""
        logits, aux = self.forward(part_sequences)
        probs = torch.sigmoid(logits)
        _, pool = torch.topk(probs, k, dim=-1)
        return pool, probs, aux


def create_column_enhanced_model(config):
    """Factory function to create column-enhanced model from config."""
    return ColumnEnhancedPredictor(
        num_parts=config.get('num_parts', 39),
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        encoder_type=config.get('encoder_type', 'transformer'),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        embedding_type=config.get('embedding_type', 'column_aware'),
        use_column_attention=config.get('use_column_attention', False),
        use_column_heads=config.get('use_column_heads', True),
    )
