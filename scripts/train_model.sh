#!/bin/bash
#SBATCH --job-name=hah-train
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=models/train_%j.log

# Run from the repo root. Set PROJECT_DIR to submit from elsewhere.
cd "${PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

echo "Starting model training..."
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

python3 src/models/train.py

echo ""
echo "Training complete: $(date)"
