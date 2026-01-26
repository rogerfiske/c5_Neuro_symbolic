"""
Hyperparameter Optimization with Optuna
========================================
Automated search for optimal model architecture and training parameters.

Search Space:
- Architecture: LSTM vs Transformer
- Embedding dimension: 32-128
- Hidden dimension: 64-256
- Number of layers: 1-4
- Learning rate: 1e-5 to 1e-2
- Dropout: 0.0-0.3
- Pool size: 20-30

Author: Dr. Synapse Research Pipeline
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import optuna

# Handle optuna-integration package (moved to separate package in newer versions)
PyTorchLightningPruningCallback = None
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
except (ImportError, ModuleNotFoundError):
    try:
        from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
    except (ImportError, ModuleNotFoundError):
        pass  # Will work without pruning callback

try:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )
except ImportError:
    plot_optimization_history = None
    plot_param_importances = None
    plot_parallel_coordinate = None
import yaml
from pathlib import Path
from datetime import datetime
import json

from train import NeuroSymbolicLightning
from data_module import CA5DataModule


def objective(trial: optuna.Trial, data_module: CA5DataModule, base_config: dict) -> float:
    """
    Optuna objective function.

    Returns Good-or-Better percentage on validation set.
    """
    # Sample hyperparameters
    config = {
        # Architecture
        'encoder_type': trial.suggest_categorical('encoder_type', ['lstm', 'transformer']),
        'embed_dim': trial.suggest_categorical('embed_dim', [32, 48, 64, 96, 128]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 192, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),

        # Training
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),

        # Sequence
        'sequence_length': trial.suggest_categorical('sequence_length', [14, 21, 30, 45, 60]),

        # Pool
        'pool_size': trial.suggest_int('pool_size', 20, 30),

        # Symbolic
        'num_rules': trial.suggest_categorical('num_rules', [10, 15, 20, 30]),
        'rule_dim': trial.suggest_categorical('rule_dim', [16, 32, 48]),
        'use_symbolic_init': trial.suggest_categorical('use_symbolic_init', [True, False]),

        # Fixed from base config
        'num_parts': base_config.get('num_parts', 39),
        'warmup_steps': base_config.get('warmup_steps', 100),
    }

    # Validate transformer config
    if config['encoder_type'] == 'transformer':
        # embed_dim must be divisible by num_heads
        while config['embed_dim'] % config['num_heads'] != 0:
            config['num_heads'] = config['num_heads'] // 2
            if config['num_heads'] < 1:
                config['num_heads'] = 1
                break

    # Update data module batch size
    data_module.batch_size = config['batch_size']
    data_module.sequence_length = config['sequence_length']

    # Create model
    model_config = {
        'num_parts': config['num_parts'],
        'embed_dim': config['embed_dim'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'encoder_type': config['encoder_type'],
        'num_heads': config['num_heads'],
        'num_rules': config['num_rules'],
        'rule_dim': config['rule_dim'],
        'dropout': config['dropout']
    }

    model = NeuroSymbolicLightning(
        model_config=model_config,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        pool_size=config['pool_size'],
        use_symbolic_init=config['use_symbolic_init']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val/good_or_better',
            patience=5,
            mode='max'
        )
    ]
    # Add pruning callback if available
    if PyTorchLightningPruningCallback is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor='val/good_or_better'))

    # Trainer (reduced epochs for hyperopt)
    trainer = pl.Trainer(
        max_epochs=30,  # Reduced for speed
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False
    )

    try:
        trainer.fit(model, data_module)
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

    # Return best validation metric
    if trainer.callback_metrics.get('val/good_or_better') is not None:
        return trainer.callback_metrics['val/good_or_better'].item()
    return 0.0


def run_hyperopt(
    n_trials: int = 100,
    data_path: str = 'data/CA5_date.csv',
    output_dir: str = 'outputs/hyperopt',
    study_name: str = 'neuro_symbolic_hyperopt',
    n_jobs: int = 1,  # Parallel trials (1 for single GPU)
    timeout_hours: float = 4.0
):
    """
    Run hyperparameter optimization.

    Args:
        n_trials: Number of optimization trials
        data_path: Path to data file
        output_dir: Output directory
        study_name: Optuna study name
        n_jobs: Parallel jobs
        timeout_hours: Maximum hours to run
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout_hours} hours")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Base configuration
    base_config = {
        'data_path': data_path,
        'num_parts': 39,
        'val_years': 0.5,
        'test_years': 2.0,
        'num_workers': 4,
        'warmup_steps': 100
    }

    # Create data module (will be reused)
    data_module = CA5DataModule(
        data_path=data_path,
        sequence_length=30,
        batch_size=64,
        num_workers=base_config['num_workers'],
        val_years=base_config['val_years'],
        test_years=base_config['test_years']
    )
    data_module.setup('fit')

    # Create Optuna study
    storage = f"sqlite:///{output_path / 'hyperopt.db'}"
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Add known good configurations as initial trials
    study.enqueue_trial({
        'encoder_type': 'transformer',
        'embed_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'sequence_length': 30,
        'pool_size': 27,
        'num_rules': 20,
        'rule_dim': 32,
        'use_symbolic_init': True
    })

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, data_module, base_config),
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}% Good-or-Better")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path / 'best_params.yaml', 'w') as f:
        yaml.dump(results, f)

    # Save all trials
    trials_data = []
    for trial in study.trials:
        if trial.value is not None:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })

    with open(output_path / 'all_trials.json', 'w') as f:
        json.dump(trials_data, f, indent=2)

    # Generate visualizations
    if plot_optimization_history is not None:
        try:
            fig = plot_optimization_history(study)
            fig.write_html(str(output_path / 'optimization_history.html'))

            fig = plot_param_importances(study)
            fig.write_html(str(output_path / 'param_importance.html'))

            fig = plot_parallel_coordinate(study)
            fig.write_html(str(output_path / 'parallel_coordinate.html'))

            print(f"\nVisualizations saved to: {output_path}")
        except Exception as e:
            print(f"Could not generate visualizations: {e}")
    else:
        print("\nVisualization libraries not available, skipping plots.")

    return study


def train_best_model(study: optuna.Study, data_path: str = 'data/CA5_date.csv'):
    """
    Train final model with best hyperparameters.
    """
    from train import train

    best_params = study.best_params

    # Build full config
    config = {
        'data_path': data_path,
        'num_parts': 39,
        'val_years': 0.5,
        'test_years': 2.0,
        'num_workers': 4,
        'max_epochs': 100,
        'patience': 15,
        'output_dir': 'outputs/best_model',
        'checkpoint_dir': 'outputs/best_model/checkpoints',
        'log_dir': 'outputs/best_model/logs',
        **best_params
    }

    print("\n" + "=" * 60)
    print("TRAINING BEST MODEL")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    return train(config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--timeout', type=float, default=4.0, help='Timeout in hours')
    parser.add_argument('--data_path', type=str, default='data/CA5_date.csv')
    parser.add_argument('--output_dir', type=str, default='outputs/hyperopt')
    parser.add_argument('--train_best', action='store_true', help='Train best model after search')
    args = parser.parse_args()

    study = run_hyperopt(
        n_trials=args.n_trials,
        data_path=args.data_path,
        output_dir=args.output_dir,
        timeout_hours=args.timeout
    )

    if args.train_best:
        train_best_model(study, args.data_path)
