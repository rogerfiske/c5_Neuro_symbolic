#!/bin/bash
# RunPod Setup Script
# ===================
# Run this after uploading and extracting the package

echo "=============================================="
echo "Neuro-Symbolic Pipeline Setup"
echo "=============================================="

# Check CUDA
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import pytorch_lightning
import optuna
print(f'PyTorch: {torch.__version__}')
print(f'Lightning: {pytorch_lightning.__version__}')
print(f'Optuna: {optuna.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/checkpoints outputs/logs outputs/hyperopt

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start Jupyter: jupyter lab"
echo "  2. Open: neuro_symbolic_pipeline.ipynb"
echo ""
echo "Or run from command line:"
echo "  python train.py --config config.yaml"
echo "  python hyperopt.py --n_trials 50"
echo "=============================================="
