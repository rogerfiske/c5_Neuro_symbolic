"""
Phase 2 Analysis Runner
=======================
Runs per-part inference and ensemble experiments in sequence.

Run on RunPod B200 with:
    python run_phase2_analysis.py

Author: Dr. Synapse Research Pipeline
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 70)
    print("PHASE 2 ANALYSIS: PER-PART INFERENCE + ENSEMBLE EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check for outputs directory structure
    base_dir = Path('.')
    outputs_dir = base_dir / 'outputs'

    # Verify best model exists
    best_model_dir = outputs_dir / 'best_model'
    if not best_model_dir.exists():
        print(f"ERROR: Best model directory not found at {best_model_dir}")
        print("Please ensure previous training outputs are present.")
        sys.exit(1)

    checkpoint_dir = best_model_dir / 'checkpoints'
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    # Step 1: Per-Part Inference Analysis
    print("\n" + "=" * 70)
    print("STEP 1: PER-PART INFERENCE ANALYSIS")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, 'per_part_inference.py'],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"WARNING: per_part_inference.py exited with code {result.returncode}")

    # Step 2: Ensemble Experiments
    print("\n" + "=" * 70)
    print("STEP 2: ENSEMBLE EXPERIMENTS")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, 'ensemble_experiment.py'],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"WARNING: ensemble_experiment.py exited with code {result.returncode}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 ANALYSIS COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # List output files
    print("\nOutput files generated:")

    per_part_dir = outputs_dir / 'per_part_analysis'
    if per_part_dir.exists():
        for f in per_part_dir.iterdir():
            print(f"  - {f}")

    ensemble_dir = outputs_dir / 'ensemble_experiment'
    if ensemble_dir.exists():
        for f in ensemble_dir.iterdir():
            print(f"  - {f}")

    print("\n" + "=" * 70)
    print("Download these files and terminate the pod when ready.")
    print("=" * 70)


if __name__ == '__main__':
    main()
