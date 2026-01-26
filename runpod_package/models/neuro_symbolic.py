"""
Neuro-Symbolic Architecture for Part Prediction
================================================
Deep learning model with symbolic rule integration.

Architecture:
- Part embeddings (learned representations)
- Temporal encoder (LSTM or Transformer)
- Symbolic attention (rules as soft constraints)
- Multi-task output (per-part probability + rule firing)

Author: Dr. Synapse Research Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PartEmbedding(nn.Module):
    """Learned embeddings for each part with positional encoding."""

    def __init__(self, num_parts=39, embed_dim=64, dropout=0.1):
        super().__init__()
        self.num_parts = num_parts
        self.embed_dim = embed_dim

        # Part embeddings
        self.part_embed = nn.Embedding(num_parts + 1, embed_dim, padding_idx=0)

        # Positional encoding for sequence position
        self.pos_embed = nn.Embedding(100, embed_dim)  # Max sequence length 100

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, part_ids, positions=None):
        """
        Args:
            part_ids: (batch, seq_len, 5) - parts used each day
            positions: optional position indices
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, parts_per_day = part_ids.shape

        # Embed each part
        part_embeds = self.part_embed(part_ids)  # (batch, seq_len, 5, embed_dim)

        # Aggregate parts per day (mean pooling)
        day_embeds = part_embeds.mean(dim=2)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        if positions is None:
            positions = torch.arange(seq_len, device=part_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embed(positions)

        # Combine
        embeddings = day_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SymbolicRuleEncoder(nn.Module):
    """Encode symbolic rules as learnable vectors."""

    def __init__(self, num_rules=20, rule_dim=32, num_parts=39):
        super().__init__()
        self.num_rules = num_rules
        self.rule_dim = rule_dim
        self.num_parts = num_parts

        # Rule embeddings
        self.rule_embed = nn.Embedding(num_rules, rule_dim)

        # Rule-to-part attention
        self.rule_part_attn = nn.Linear(rule_dim, num_parts)

        # Rule confidence (learnable)
        self.rule_confidence = nn.Parameter(torch.ones(num_rules) * 0.5)

    def forward(self, rule_ids=None):
        """
        Returns:
            rule_vectors: (num_rules, rule_dim)
            rule_part_weights: (num_rules, num_parts) - how much each rule affects each part
        """
        if rule_ids is None:
            rule_ids = torch.arange(self.num_rules, device=self.rule_embed.weight.device)

        rule_vectors = self.rule_embed(rule_ids)
        rule_part_weights = torch.sigmoid(self.rule_part_attn(rule_vectors))

        # Scale by learned confidence
        confidences = torch.sigmoid(self.rule_confidence[rule_ids]).unsqueeze(-1)
        rule_part_weights = rule_part_weights * confidences

        return rule_vectors, rule_part_weights


class TemporalEncoder(nn.Module):
    """LSTM or Transformer encoder for temporal sequences."""

    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2,
                 encoder_type='lstm', num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim

        if encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.output_dim = hidden_dim * 2  # Bidirectional

        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = input_dim

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: optional attention mask
        Returns:
            encoded: (batch, seq_len, output_dim)
        """
        if self.encoder_type == 'lstm':
            encoded, _ = self.encoder(x)
        else:
            encoded = self.encoder(x, src_key_padding_mask=mask)

        return self.layer_norm(encoded)


class SymbolicAttention(nn.Module):
    """Attention mechanism that incorporates symbolic rules."""

    def __init__(self, context_dim=256, rule_dim=32, num_parts=39, num_heads=4):
        super().__init__()
        self.num_parts = num_parts
        self.num_heads = num_heads

        # Query: what parts are we predicting?
        self.query_proj = nn.Linear(context_dim, num_parts * num_heads)

        # Key: rule representations
        self.key_proj = nn.Linear(rule_dim, num_parts * num_heads)

        # Value: rule effects on parts
        self.value_proj = nn.Linear(rule_dim, num_parts * num_heads)

        # Output projection
        self.out_proj = nn.Linear(num_parts * num_heads, num_parts)

        self.scale = math.sqrt(num_parts)

    def forward(self, context, rule_vectors, rule_part_weights):
        """
        Multi-head attention over symbolic rules.

        Args:
            context: (batch, context_dim) - temporal context from encoder
            rule_vectors: (num_rules, rule_dim) - learned rule embeddings
            rule_part_weights: (num_rules, num_parts) - rule-to-part influence weights
        Returns:
            rule_adjustments: (batch, num_parts) - per-part score adjustments
        """
        batch_size = context.shape[0]
        num_rules = rule_vectors.shape[0]
        device = context.device

        # Project context to queries: (batch, num_heads, num_parts)
        Q = self.query_proj(context)  # (batch, num_parts * num_heads)
        Q = Q.view(batch_size, self.num_heads, self.num_parts)

        # Project rules to keys and values: (num_rules, num_heads, num_parts)
        K = self.key_proj(rule_vectors)  # (num_rules, num_parts * num_heads)
        K = K.view(num_rules, self.num_heads, self.num_parts)

        V = self.value_proj(rule_vectors)  # (num_rules, num_parts * num_heads)
        V = V.view(num_rules, self.num_heads, self.num_parts)

        # Compute attention scores via batched matrix multiplication
        # Q: (batch, heads, parts) -> transpose to (batch, heads, parts)
        # K: (rules, heads, parts) -> need (heads, parts, rules) for matmul
        K_t = K.permute(1, 2, 0)  # (heads, parts, rules)

        # Expand K for batch: (1, heads, parts, rules) -> (batch, heads, parts, rules)
        K_t = K_t.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Attention: Q @ K^T over parts dimension
        # Q: (batch, heads, parts), K_t: (batch, heads, parts, rules)
        # Result: (batch, heads, rules)
        attn_scores = torch.matmul(Q.unsqueeze(2), K_t).squeeze(2) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, rules)

        # Apply attention to values
        # V: (rules, heads, parts) -> (heads, rules, parts)
        V_t = V.permute(1, 0, 2)  # (heads, rules, parts)
        V_t = V_t.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, heads, rules, parts)

        # attn_weights: (batch, heads, rules) @ V_t: (batch, heads, rules, parts)
        # Result: (batch, heads, parts)
        rule_effects = torch.matmul(attn_weights.unsqueeze(2), V_t).squeeze(2)

        # Combine heads: (batch, heads * parts)
        rule_effects = rule_effects.reshape(batch_size, self.num_heads * self.num_parts)

        # Final projection to part space
        adjustments = self.out_proj(rule_effects)  # (batch, num_parts)

        return adjustments


class NeuroSymbolicPredictor(nn.Module):
    """
    Full neuro-symbolic model for part prediction.

    Combines:
    - Deep temporal encoding (LSTM/Transformer)
    - Symbolic rule integration via attention
    - Multi-task learning (prediction + rule firing)
    """

    def __init__(
        self,
        num_parts=39,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        encoder_type='transformer',
        num_heads=4,
        num_rules=20,
        rule_dim=32,
        dropout=0.1
    ):
        super().__init__()
        self.num_parts = num_parts

        # Part embedding layer
        self.part_embedding = PartEmbedding(
            num_parts=num_parts,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_type=encoder_type,
            num_heads=num_heads,
            dropout=dropout
        )

        # Symbolic rule encoder
        self.rule_encoder = SymbolicRuleEncoder(
            num_rules=num_rules,
            rule_dim=rule_dim,
            num_parts=num_parts
        )

        # Context aggregation
        context_dim = self.temporal_encoder.output_dim
        self.context_pool = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Symbolic attention
        self.symbolic_attention = SymbolicAttention(
            context_dim=hidden_dim,
            rule_dim=rule_dim,
            num_parts=num_parts,
            num_heads=num_heads
        )

        # Neural prediction head
        self.neural_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_parts)
        )

        # Fusion gate (learnable balance between neural and symbolic)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Stability regularizer (penalize large changes)
        self.stability_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, part_sequences, previous_pool=None):
        """
        Args:
            part_sequences: (batch, seq_len, 5) - historical part usage
            previous_pool: (batch, num_parts) - binary mask of previous prediction
        Returns:
            logits: (batch, num_parts) - prediction scores
            aux_outputs: dict with intermediate values for analysis
        """
        batch_size = part_sequences.shape[0]

        # Embed sequences
        embeddings = self.part_embedding(part_sequences)  # (batch, seq_len, embed_dim)

        # Temporal encoding
        encoded = self.temporal_encoder(embeddings)  # (batch, seq_len, hidden_dim)

        # Pool to get context (use last timestep or mean)
        context = encoded[:, -1, :]  # (batch, hidden_dim)
        context = self.context_pool(context)  # (batch, hidden_dim)

        # Neural prediction
        neural_logits = self.neural_head(context)  # (batch, num_parts)

        # Symbolic rule integration
        rule_vectors, rule_part_weights = self.rule_encoder()
        rule_adjustments = self.symbolic_attention(context, rule_vectors, rule_part_weights)

        # Fusion: combine neural and symbolic
        gate = self.fusion_gate(context)  # (batch, 1)
        fused_logits = gate * neural_logits + (1 - gate) * rule_adjustments

        # Stability regularization
        if previous_pool is not None:
            stability_bonus = previous_pool * torch.sigmoid(self.stability_weight)
            fused_logits = fused_logits + stability_bonus

        aux_outputs = {
            'neural_logits': neural_logits,
            'rule_adjustments': rule_adjustments,
            'fusion_gate': gate,
            'rule_part_weights': rule_part_weights,
            'context': context
        }

        return fused_logits, aux_outputs

    def predict_pool(self, part_sequences, k=27, previous_pool=None):
        """
        Make pool prediction.

        Args:
            part_sequences: (batch, seq_len, 5)
            k: pool size
            previous_pool: optional previous prediction
        Returns:
            pool: (batch, k) - indices of selected parts
            probs: (batch, num_parts) - probability of each part
        """
        logits, aux = self.forward(part_sequences, previous_pool)
        probs = torch.sigmoid(logits)

        # Select top-k
        _, pool = torch.topk(probs, k, dim=-1)

        return pool, probs, aux


class NeuroSymbolicLoss(nn.Module):
    """
    Multi-task loss for neuro-symbolic training.

    Components:
    - Binary cross-entropy for part prediction
    - Rule consistency loss
    - Stability penalty
    - Tier-aware weighting
    """

    def __init__(self, num_parts=39, tier_weights=None):
        super().__init__()
        self.num_parts = num_parts

        # Default tier weights (prioritize getting 4-5 hits)
        if tier_weights is None:
            tier_weights = {'excellent': 2.0, 'good': 1.5, 'acceptable': 1.0}
        self.tier_weights = tier_weights

        # Focal loss for class imbalance
        self.focal_gamma = 2.0

    def forward(self, logits, targets, aux_outputs=None, previous_targets=None):
        """
        Args:
            logits: (batch, num_parts) - predicted scores
            targets: (batch, num_parts) - binary targets (1 if part used)
            aux_outputs: dict from model forward
            previous_targets: optional previous day's targets for stability
        Returns:
            loss: scalar
            loss_components: dict with individual losses
        """
        # Focal BCE loss
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal weighting
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = (focal_weight * bce).mean()

        loss_components = {'focal_bce': focal_loss.item()}
        total_loss = focal_loss

        # Rule consistency loss
        if aux_outputs is not None and 'rule_part_weights' in aux_outputs:
            rule_weights = aux_outputs['rule_part_weights']
            # Rules should be sparse and confident
            rule_entropy = -(rule_weights * torch.log(rule_weights + 1e-8) +
                            (1 - rule_weights) * torch.log(1 - rule_weights + 1e-8))
            rule_loss = rule_entropy.mean() * 0.01
            total_loss = total_loss + rule_loss
            loss_components['rule_entropy'] = rule_loss.item()

        # Stability loss
        if previous_targets is not None:
            # Penalize predicting different parts than yesterday
            stability_loss = F.mse_loss(probs, previous_targets) * 0.1
            total_loss = total_loss + stability_loss
            loss_components['stability'] = stability_loss.item()

        loss_components['total'] = total_loss.item()
        return total_loss, loss_components


def create_model(config):
    """Factory function to create model from config dict."""
    return NeuroSymbolicPredictor(
        num_parts=config.get('num_parts', 39),
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        encoder_type=config.get('encoder_type', 'transformer'),
        num_heads=config.get('num_heads', 4),
        num_rules=config.get('num_rules', 20),
        rule_dim=config.get('rule_dim', 32),
        dropout=config.get('dropout', 0.1)
    )
