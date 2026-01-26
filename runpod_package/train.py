"""
Training Script for Neuro-Symbolic Part Prediction
==================================================
Full training pipeline with PyTorch Lightning.

Features:
- Mixed precision training (FP16/BF16 on H200)
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard/WandB logging

Author: Dr. Synapse Research Pipeline
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import argparse

from models import NeuroSymbolicPredictor, NeuroSymbolicLoss, create_model
from data_module import CA5DataModule, load_symbolic_rules, initialize_rule_encoder


class NeuroSymbolicLightning(pl.LightningModule):
    """
    PyTorch Lightning module for neuro-symbolic training.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 100,
        pool_size: int = 27,
        use_symbolic_init: bool = True,
        rule_path: Optional[str] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = create_model(model_config)
        self.loss_fn = NeuroSymbolicLoss(num_parts=model_config.get('num_parts', 39))

        # Config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.pool_size = pool_size

        # Initialize with symbolic rules
        if use_symbolic_init:
            rules = load_symbolic_rules(rule_path)
            initialize_rule_encoder(self.model, rules)

        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, batch):
        """Forward pass."""
        sequence = batch['sequence']
        prev_target = batch.get('prev_target', None)
        logits, aux = self.model(sequence, prev_target)
        return logits, aux

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, aux = self.forward(batch)
        target = batch['target']
        prev_target = batch.get('prev_target', None)

        loss, loss_components = self.loss_fn(logits, target, aux, prev_target)

        # Log losses
        self.log('train/loss', loss, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'train/{name}', value)

        # Log fusion gate (how much neural vs symbolic)
        if 'fusion_gate' in aux:
            self.log('train/fusion_gate', aux['fusion_gate'].mean())

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with tier metrics."""
        logits, aux = self.forward(batch)
        target = batch['target']

        loss, _ = self.loss_fn(logits, target, aux)
        self.log('val/loss', loss, prog_bar=True)

        # Compute tier metrics
        probs = torch.sigmoid(logits)
        pool = torch.topk(probs, self.pool_size, dim=-1).indices

        # Count hits
        batch_size = target.shape[0]
        hits = []
        for i in range(batch_size):
            predicted_parts = set(pool[i].cpu().numpy() + 1)  # 1-indexed
            actual_parts = set((target[i] > 0.5).nonzero().squeeze(-1).cpu().numpy() + 1)
            hits.append(len(predicted_parts & actual_parts))

        self.validation_outputs.append({
            'loss': loss.item(),
            'hits': hits
        })

        return loss

    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics."""
        if not self.validation_outputs:
            return

        all_hits = []
        for output in self.validation_outputs:
            all_hits.extend(output['hits'])

        all_hits = np.array(all_hits)

        # Tier breakdown
        excellent = (all_hits == 5).mean() * 100
        good = (all_hits == 4).mean() * 100
        good_or_better = excellent + good

        self.log('val/excellent_pct', excellent)
        self.log('val/good_pct', good)
        self.log('val/good_or_better', good_or_better, prog_bar=True)
        self.log('val/avg_hits', all_hits.mean())

        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, aux = self.forward(batch)
        target = batch['target']
        prev_target = batch.get('prev_target', None)

        loss, _ = self.loss_fn(logits, target, aux)

        # Compute predictions
        probs = torch.sigmoid(logits)
        pool = torch.topk(probs, self.pool_size, dim=-1).indices

        # Compute stability (Jaccard with previous pool)
        if prev_target is not None:
            prev_pool = torch.topk(prev_target, self.pool_size, dim=-1).indices
            jaccard = self._compute_jaccard(pool, prev_pool)
        else:
            jaccard = torch.zeros(pool.shape[0])

        # Count hits
        batch_size = target.shape[0]
        hits = []
        for i in range(batch_size):
            predicted_parts = set(pool[i].cpu().numpy() + 1)
            actual_parts = set((target[i] > 0.5).nonzero().squeeze(-1).cpu().numpy() + 1)
            hits.append(len(predicted_parts & actual_parts))

        self.test_outputs.append({
            'loss': loss.item(),
            'hits': hits,
            'jaccard': jaccard.mean().item() if torch.is_tensor(jaccard) else jaccard,
            'fusion_gate': aux['fusion_gate'].mean().item() if 'fusion_gate' in aux else 0.5
        })

        return loss

    def _compute_jaccard(self, pool1, pool2):
        """Compute Jaccard similarity between two pools."""
        batch_size = pool1.shape[0]
        jaccard = []
        for i in range(batch_size):
            set1 = set(pool1[i].cpu().numpy())
            set2 = set(pool2[i].cpu().numpy())
            if len(set1 | set2) > 0:
                jaccard.append(len(set1 & set2) / len(set1 | set2))
            else:
                jaccard.append(0.0)
        return torch.tensor(jaccard)

    def on_test_epoch_end(self):
        """Compute final test metrics."""
        if not self.test_outputs:
            return

        all_hits = []
        all_jaccard = []
        all_fusion = []

        for output in self.test_outputs:
            all_hits.extend(output['hits'])
            all_jaccard.append(output['jaccard'])
            all_fusion.append(output['fusion_gate'])

        all_hits = np.array(all_hits)

        # Tier breakdown
        excellent = (all_hits == 5).mean() * 100
        good = (all_hits == 4).mean() * 100
        unacceptable = (all_hits <= 3).mean() * 100
        good_or_better = excellent + good

        avg_jaccard = np.mean(all_jaccard)
        avg_fusion = np.mean(all_fusion)

        self.log('test/excellent_pct', excellent)
        self.log('test/good_pct', good)
        self.log('test/unacceptable_pct', unacceptable)
        self.log('test/good_or_better', good_or_better)
        self.log('test/avg_hits', all_hits.mean())
        self.log('test/avg_jaccard', avg_jaccard)
        self.log('test/avg_fusion_gate', avg_fusion)

        # Store final results for later retrieval
        self.final_results = {
            'excellent': excellent,
            'good': good,
            'unacceptable': unacceptable,
            'good_or_better': good_or_better,
            'avg_hits': all_hits.mean(),
            'avg_jaccard': avg_jaccard,
            'avg_fusion': avg_fusion
        }

        self.test_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * (step - self.warmup_steps) / 10000))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def train(config: Dict[str, Any]):
    """
    Main training function.

    Args:
        config: Configuration dictionary with all hyperparameters
    """
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # Data module
    data_module = CA5DataModule(
        data_path=config.get('data_path', 'data/CA5_date.csv'),
        sequence_length=config.get('sequence_length', 30),
        batch_size=config.get('batch_size', 64),
        num_workers=config.get('num_workers', 4),
        val_years=config.get('val_years', 0.5),
        test_years=config.get('test_years', 2.0)
    )

    # Model configuration
    model_config = {
        'num_parts': config.get('num_parts', 39),
        'embed_dim': config.get('embed_dim', 64),
        'hidden_dim': config.get('hidden_dim', 128),
        'num_layers': config.get('num_layers', 2),
        'encoder_type': config.get('encoder_type', 'transformer'),
        'num_heads': config.get('num_heads', 4),
        'num_rules': config.get('num_rules', 20),
        'rule_dim': config.get('rule_dim', 32),
        'dropout': config.get('dropout', 0.1)
    }

    # Lightning module
    model = NeuroSymbolicLightning(
        model_config=model_config,
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        warmup_steps=config.get('warmup_steps', 100),
        pool_size=config.get('pool_size', 27),
        use_symbolic_init=config.get('use_symbolic_init', True),
        rule_path=config.get('rule_path', None)
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.get('checkpoint_dir', 'outputs/checkpoints'),
            filename='best-{epoch}-{val/good_or_better:.2f}',
            monitor='val/good_or_better',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/good_or_better',
            patience=config.get('patience', 10),
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=10)
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=config.get('log_dir', 'outputs/logs'),
        name='neuro_symbolic'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator='auto',
        devices=1,
        precision=config.get('precision', '16-mixed'),  # FP16 for speed
        gradient_clip_val=config.get('gradient_clip', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad', 1),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=config.get('val_check_interval', 1.0),
        enable_progress_bar=True,
        deterministic=False  # Faster on GPU
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, data_module)

    # Test
    print("\nRunning final test...")
    trainer.test(model, data_module, ckpt_path='best', weights_only=False)

    # Save final config
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"\nOutputs saved to: {output_dir}")

    return trainer, model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Neuro-Symbolic Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--data_path', type=str, default='data/CA5_date.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--encoder_type', type=str, default='transformer', choices=['lstm', 'transformer'])
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--pool_size', type=int, default=27)
    parser.add_argument('--sequence_length', type=int, default=30)
    args = parser.parse_args()

    # Load config from file or use defaults
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = vars(args)

    train(config)


if __name__ == '__main__':
    main()
