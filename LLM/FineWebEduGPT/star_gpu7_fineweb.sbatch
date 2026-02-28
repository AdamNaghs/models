#!/bin/bash
#SBATCH --job-name=fineweb-train
#SBATCH --output=finweb-train.out
#SBATCH --error=finweb-train.err
#SBATCH --nodes=1
#SBATCH --nodelist=gpu7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00

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
torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py --cache-gb 500 --eval-every 5 --eval-iters 12 --batch-size 64 --context 1024 --grad-accum 4 --num-workers 8 --queue-size 25
