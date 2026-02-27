#!/bin/bash
#SBATCH --job-name=fineweb-h100
#SBATCH --output=finweb-h100.out
#SBATCH --error=finweb-h100.err
#SBATCH --nodes=1
#SBATCH --nodelist=gpu7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Optional (uncomment if your cluster requires them)
##SBATCH --partition=gpu
##SBATCH --qos=normal
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@hofstra.edu

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Activate your environment (edit if needed)
source llmvenv/bin/activate

pip install -r requirements.txt

# Helpful runtime diagnostics
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Multi-GPU launch (single node)
# Preset options: 5080 | hpc | 10b | a100 | h100
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train_fineweb_gpt.py \
  --preset h100 \
  --config CC-MAIN-2025-26
