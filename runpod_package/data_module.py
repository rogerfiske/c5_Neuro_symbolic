"""
Data Loading and Preprocessing for Neuro-Symbolic Pipeline
==========================================================
Handles CA5 dataset loading, sequence generation, and PyTorch DataLoaders.

Author: Dr. Synapse Research Pipeline
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import timedelta


class CA5Dataset(Dataset):
    """
    PyTorch Dataset for CA5 part prediction.

    Generates sequences of historical part usage for predicting next day.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 30,
        num_parts: int = 39,
        include_features: bool = True
    ):
        """
        Args:
            data: DataFrame with columns ['date', 'm_1', 'm_2', 'm_3', 'm_4', 'm_5']
            sequence_length: Number of historical days to use
            num_parts: Total number of unique parts
            include_features: Whether to compute additional features
        """
        self.sequence_length = sequence_length
        self.num_parts = num_parts
        self.include_features = include_features

        # Sort by date
        self.data = data.sort_values('date').reset_index(drop=True)
        self.dates = self.data['date'].values

        # Pre-extract part arrays for speed
        self.parts = self.data[['m_1', 'm_2', 'm_3', 'm_4', 'm_5']].values.astype(np.int64)

        # Pre-compute binary targets (multi-hot encoding)
        self.targets = np.zeros((len(self.data), num_parts), dtype=np.float32)
        for i, row in enumerate(self.parts):
            for p in row:
                if 1 <= p <= num_parts:
                    self.targets[i, p - 1] = 1.0

        # Valid indices (need sequence_length history + 1 for target)
        self.valid_indices = list(range(sequence_length, len(self.data) - 1))

        # Pre-compute additional features if requested
        if include_features:
            self._compute_features()

    def _compute_features(self):
        """Compute additional engineered features."""
        n = len(self.data)

        # Time since last use (TSLU) for each part
        self.tslu = np.zeros((n, self.num_parts), dtype=np.float32)
        last_seen = np.full(self.num_parts, -self.sequence_length * 2)

        for i in range(n):
            for p in self.parts[i]:
                if 1 <= p <= self.num_parts:
                    last_seen[p - 1] = i
            for p in range(self.num_parts):
                self.tslu[i, p] = min((i - last_seen[p]) / self.sequence_length, 2.0)

        # Rolling frequency (7, 14, 30 days)
        self.rolling_freq = {}
        for window in [7, 14, 30]:
            freq = np.zeros((n, self.num_parts), dtype=np.float32)
            for i in range(window, n):
                for j in range(i - window, i):
                    for p in self.parts[j]:
                        if 1 <= p <= self.num_parts:
                            freq[i, p - 1] += 1
                freq[i] = freq[i] / window
            self.rolling_freq[window] = freq

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: (seq_len, 5) - historical part IDs
            target: (num_parts,) - binary target for next day
            features: dict of additional features (if enabled)
        """
        actual_idx = self.valid_indices[idx]

        # Historical sequence
        start_idx = actual_idx - self.sequence_length
        sequence = self.parts[start_idx:actual_idx].copy()

        # Target (next day's parts)
        target = self.targets[actual_idx].copy()

        # Previous day's target (for stability)
        prev_target = self.targets[actual_idx - 1].copy()

        result = {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32),
            'prev_target': torch.tensor(prev_target, dtype=torch.float32),
            'date_idx': actual_idx
        }

        if self.include_features:
            result['tslu'] = torch.tensor(self.tslu[actual_idx - 1], dtype=torch.float32)
            result['freq_7'] = torch.tensor(self.rolling_freq[7][actual_idx - 1], dtype=torch.float32)
            result['freq_14'] = torch.tensor(self.rolling_freq[14][actual_idx - 1], dtype=torch.float32)
            result['freq_30'] = torch.tensor(self.rolling_freq[30][actual_idx - 1], dtype=torch.float32)

        return result


class CA5DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CA5 dataset.

    Handles train/val/test splits and DataLoader creation.
    """

    def __init__(
        self,
        data_path: str = 'data/CA5_date.csv',
        sequence_length: int = 30,
        batch_size: int = 64,
        num_workers: int = 4,
        val_years: float = 0.5,
        test_years: float = 2.0,
        num_parts: int = 39
    ):
        """
        Args:
            data_path: Path to CA5_date.csv
            sequence_length: Historical sequence length
            batch_size: Batch size for training
            num_workers: DataLoader workers
            val_years: Years of data for validation
            test_years: Years of data for testing
            num_parts: Number of unique parts
        """
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_years = val_years
        self.test_years = test_years
        self.num_parts = num_parts

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download or check data exists."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test datasets."""
        # Load data
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Calculate split points
        last_date = df['date'].max()
        test_cutoff = last_date - timedelta(days=int(365 * self.test_years))
        val_cutoff = test_cutoff - timedelta(days=int(365 * self.val_years))

        # Split data
        train_df = df[df['date'] < val_cutoff].copy()
        val_df = df[(df['date'] >= val_cutoff) & (df['date'] < test_cutoff)].copy()
        test_df = df[df['date'] >= test_cutoff].copy()

        print(f"Data splits:")
        print(f"  Train: {len(train_df)} days ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"  Val:   {len(val_df)} days ({val_df['date'].min()} to {val_df['date'].max()})")
        print(f"  Test:  {len(test_df)} days ({test_df['date'].min()} to {test_df['date'].max()})")

        # For training, we need overlapping sequences, so include some history
        # Merge train+val for creating sequences that span the boundary
        train_val_df = df[df['date'] < test_cutoff].copy()

        if stage == 'fit' or stage is None:
            # Training dataset (use train_val for sequences, but only train indices)
            self.train_dataset = CA5Dataset(
                train_df,
                sequence_length=self.sequence_length,
                num_parts=self.num_parts,
                include_features=True
            )

            # Validation dataset
            self.val_dataset = CA5Dataset(
                train_val_df,
                sequence_length=self.sequence_length,
                num_parts=self.num_parts,
                include_features=True
            )
            # Filter to only validation period
            val_start_idx = len(train_df)
            self.val_dataset.valid_indices = [
                i for i in self.val_dataset.valid_indices
                if i >= val_start_idx
            ]

        if stage == 'test' or stage is None:
            # Test dataset (use all data for sequences)
            self.test_dataset = CA5Dataset(
                df,
                sequence_length=self.sequence_length,
                num_parts=self.num_parts,
                include_features=True
            )
            # Filter to only test period
            test_start_idx = len(train_df) + len(val_df)
            self.test_dataset.valid_indices = [
                i for i in self.test_dataset.valid_indices
                if i >= test_start_idx
            ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )


def load_symbolic_rules(rule_path: Optional[str] = None) -> List[dict]:
    """
    Load symbolic rules from CSV file.

    Returns list of rule dicts with keys:
    - id, type, description, confidence, lift
    - antecedent, consequent (for sequential rules)
    - part (for burst rules)
    """
    rules = []

    if rule_path is None or not Path(rule_path).exists():
        # Default rules based on previous analysis
        default_rules = [
            {'id': 'SEQ_001', 'type': 'sequential', 'antecedent': 3, 'consequent': 36, 'lift': 1.186, 'confidence': 0.15},
            {'id': 'SEQ_002', 'type': 'sequential', 'antecedent': 14, 'consequent': 27, 'lift': 1.175, 'confidence': 0.15},
            {'id': 'SEQ_003', 'type': 'sequential', 'antecedent': 39, 'consequent': 30, 'lift': 1.164, 'confidence': 0.15},
            {'id': 'SEQ_004', 'type': 'sequential', 'antecedent': 9, 'consequent': 3, 'lift': 1.164, 'confidence': 0.15},
            {'id': 'SEQ_005', 'type': 'sequential', 'antecedent': 9, 'consequent': 34, 'lift': 1.163, 'confidence': 0.15},
            {'id': 'BURST_001', 'type': 'burst', 'part': 13, 'lift': 1.136, 'confidence': 0.15},
        ]
        return default_rules

    rule_df = pd.read_csv(rule_path)
    for _, row in rule_df.iterrows():
        rule = {
            'id': row['rule_id'],
            'type': row['type'],
            'confidence': row['confidence'],
            'lift': row['lift']
        }

        desc = row['description']
        if 'today ->' in desc:
            parts = desc.replace('Part ', '').replace(' today', '').replace(' tomorrow', '').split(' -> ')
            rule['antecedent'] = int(parts[0])
            rule['consequent'] = int(parts[1])
        elif 'consecutive' in desc:
            rule['part'] = int(desc.split('Part ')[1].split(' ')[0])

        rules.append(rule)

    return rules


def initialize_rule_encoder(model, rules: List[dict]):
    """
    Initialize rule encoder weights based on discovered rules.

    This injects symbolic knowledge into the neural network.
    """
    num_rules = model.rule_encoder.num_rules
    num_parts = model.rule_encoder.num_parts

    # Initialize rule-part attention based on discovered rules
    with torch.no_grad():
        # Reset to small random values
        model.rule_encoder.rule_part_attn.weight.data.normal_(0, 0.01)
        model.rule_encoder.rule_part_attn.bias.data.zero_()

        for i, rule in enumerate(rules[:num_rules]):
            if rule['type'] == 'sequential':
                # Boost consequent when antecedent present
                cons = rule['consequent'] - 1  # 0-indexed
                model.rule_encoder.rule_part_attn.bias.data[cons] += rule['lift'] - 1

            elif rule['type'] == 'burst':
                # Boost part repeating
                part = rule['part'] - 1
                model.rule_encoder.rule_part_attn.bias.data[part] += rule['lift'] - 1

            # Set confidence
            model.rule_encoder.rule_confidence.data[i] = rule['confidence']

    print(f"Initialized rule encoder with {len(rules)} symbolic rules")
