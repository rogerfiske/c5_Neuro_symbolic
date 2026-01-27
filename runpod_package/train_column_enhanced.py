"""
Column-Enhanced Model Training
==============================
Trains and evaluates column-enhanced neural models.

Tests multiple configurations:
1. Standard embeddings (baseline)
2. Column-aware embeddings
3. Column-feature embeddings
4. Per-column output heads
5. All enhancements combined

Run on RunPod with:
    python train_column_enhanced.py

Author: Dr. Synapse Research Pipeline
Date: 2026-01-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
from collections import defaultdict

from models.column_enhanced import create_column_enhanced_model
from data_module import CA5DataModule


class ColumnEnhancedModule(pl.LightningModule):
    """PyTorch Lightning module for column-enhanced training."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Create model
        self.model = create_column_enhanced_model(config)

        # Loss function
        self.focal_gamma = config.get('focal_gamma', 2.0)

        # Pool size for evaluation
        self.pool_size = config.get('pool_size', 30)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, targets):
        """Focal BCE loss."""
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal weighting
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        loss = (focal_weight * bce).mean()

        return loss

    def compute_metrics(self, probs, targets):
        """Compute tier metrics."""
        batch_size = probs.shape[0]

        # Get top-k predictions
        _, top_k = torch.topk(probs, self.pool_size, dim=-1)

        # Create prediction mask
        pred_mask = torch.zeros_like(probs)
        pred_mask.scatter_(1, top_k, 1.0)

        # Count hits
        hits = (pred_mask * targets).sum(dim=-1)

        # Tier counts
        excellent = (hits == 5).sum().item()
        good = (hits == 4).sum().item()
        unacceptable = (hits <= 3).sum().item()

        return {
            'hits_mean': hits.mean().item(),
            'excellent': excellent,
            'good': good,
            'unacceptable': unacceptable,
            'gob': excellent + good,
            'total': batch_size
        }

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        targets = batch['target']

        logits, aux = self.model(sequences)
        loss = self.compute_loss(logits, targets)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        targets = batch['target']

        logits, aux = self.model(sequences)
        loss = self.compute_loss(logits, targets)

        probs = torch.sigmoid(logits)
        metrics = self.compute_metrics(probs, targets)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_hits', metrics['hits_mean'], prog_bar=True)

        return {'loss': loss, 'metrics': metrics}

    def validation_epoch_end(self, outputs):
        # Aggregate metrics
        total = sum(o['metrics']['total'] for o in outputs)
        excellent = sum(o['metrics']['excellent'] for o in outputs)
        good = sum(o['metrics']['good'] for o in outputs)
        gob = excellent + good

        gob_pct = gob / total * 100 if total > 0 else 0

        self.log('val_gob_pct', gob_pct, prog_bar=True)
        self.log('val_excellent_pct', excellent / total * 100 if total > 0 else 0)
        self.log('val_good_pct', good / total * 100 if total > 0 else 0)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # Same aggregation as validation
        total = sum(o['metrics']['total'] for o in outputs)
        excellent = sum(o['metrics']['excellent'] for o in outputs)
        good = sum(o['metrics']['good'] for o in outputs)
        unacceptable = sum(o['metrics']['unacceptable'] for o in outputs)
        gob = excellent + good

        self.log('test_gob_pct', gob / total * 100)
        self.log('test_excellent_pct', excellent / total * 100)
        self.log('test_good_pct', good / total * 100)
        self.log('test_unacceptable_pct', unacceptable / total * 100)

        print(f"\n{'='*60}")
        print(f"TEST RESULTS")
        print(f"{'='*60}")
        print(f"Excellent: {excellent}/{total} ({excellent/total*100:.1f}%)")
        print(f"Good: {good}/{total} ({good/total*100:.1f}%)")
        print(f"GoB: {gob}/{total} ({gob/total*100:.1f}%)")
        print(f"Unacceptable: {unacceptable}/{total} ({unacceptable/total*100:.1f}%)")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('max_epochs', 50),
            eta_min=1e-6
        )

        return [optimizer], [scheduler]


def run_experiment(config, data_path, output_dir):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}")

    # Setup data
    datamodule = CA5DataModule(
        data_path=str(data_path),
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size'],
        num_workers=4,
        val_years=config.get('val_years', 0.5),
        test_years=config.get('test_years', 2.0),
        num_parts=config.get('num_parts', 39)
    )

    # Create model
    model = ColumnEnhancedModule(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / config['name'] / 'checkpoints',
        filename='best-{epoch:02d}-{val_gob_pct:.2f}',
        monitor='val_gob_pct',
        mode='max',
        save_top_k=1
    )

    early_stop = EarlyStopping(
        monitor='val_gob_pct',
        patience=10,
        mode='max'
    )

    logger = CSVLogger(output_dir, name=config['name'])

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 50),
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, datamodule)

    # Test with best checkpoint
    best_path = checkpoint_callback.best_model_path
    if best_path:
        model = ColumnEnhancedModule.load_from_checkpoint(best_path, config=config)

    results = trainer.test(model, datamodule)

    return results[0] if results else {}


def main():
    print("=" * 70)
    print("COLUMN-ENHANCED MODEL EXPERIMENTS")
    print("=" * 70)

    # Setup paths
    base_dir = Path('.')
    data_path = base_dir / 'data' / 'CA5_date.csv'
    output_dir = base_dir / 'outputs' / 'column_enhanced'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration
    base_config = {
        'num_parts': 39,
        'embed_dim': 128,
        'hidden_dim': 192,
        'num_layers': 3,
        'encoder_type': 'transformer',
        'num_heads': 2,
        'dropout': 0.2,
        'sequence_length': 14,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_epochs': 50,
        'pool_size': 30,
        'val_years': 0.5,
        'test_years': 2.0,
        'focal_gamma': 2.0,
    }

    # Experiment configurations
    experiments = [
        # Baseline: Standard embeddings, no column enhancements
        {
            **base_config,
            'name': 'baseline_standard',
            'embedding_type': 'standard',
            'use_column_attention': False,
            'use_column_heads': False,
        },

        # Approach 1: Column-aware embeddings only
        {
            **base_config,
            'name': 'column_aware_embed',
            'embedding_type': 'column_aware',
            'use_column_attention': False,
            'use_column_heads': False,
        },

        # Approach 3: Column-feature embeddings only
        {
            **base_config,
            'name': 'column_features_embed',
            'embedding_type': 'column_features',
            'use_column_attention': False,
            'use_column_heads': False,
        },

        # Approach 5: Per-column output heads only (with standard embed)
        {
            **base_config,
            'name': 'column_output_heads',
            'embedding_type': 'standard',
            'use_column_attention': False,
            'use_column_heads': True,
        },

        # Combined: Column-aware embedding + column output heads
        {
            **base_config,
            'name': 'column_aware_with_heads',
            'embedding_type': 'column_aware',
            'use_column_attention': False,
            'use_column_heads': True,
        },

        # Combined: Column-features embedding + column output heads
        {
            **base_config,
            'name': 'column_features_with_heads',
            'embedding_type': 'column_features',
            'use_column_attention': False,
            'use_column_heads': True,
        },
    ]

    # Run experiments
    all_results = []

    for exp_config in experiments:
        try:
            results = run_experiment(exp_config, data_path, output_dir)
            results['name'] = exp_config['name']
            results['embedding_type'] = exp_config['embedding_type']
            results['use_column_heads'] = exp_config['use_column_heads']
            all_results.append(results)
        except Exception as e:
            print(f"Error in {exp_config['name']}: {e}")
            all_results.append({
                'name': exp_config['name'],
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n| Configuration | Excellent | Good | GoB | Unacceptable |")
    print("|---------------|-----------|------|-----|--------------|")

    for result in all_results:
        if 'error' in result:
            print(f"| {result['name']:<30} | ERROR: {result['error'][:30]} |")
        else:
            exc = result.get('test_excellent_pct', 0)
            good = result.get('test_good_pct', 0)
            gob = result.get('test_gob_pct', 0)
            unacc = result.get('test_unacceptable_pct', 0)
            print(f"| {result['name']:<30} | {exc:>8.1f}% | {good:>4.1f}% | {gob:>3.1f}% | {unacc:>11.1f}% |")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'experiment_results.csv', index=False)

    # Find best
    valid_results = [r for r in all_results if 'test_gob_pct' in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get('test_gob_pct', 0))
        print(f"\n**Best Configuration**: {best['name']} with {best['test_gob_pct']:.1f}% GoB")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
