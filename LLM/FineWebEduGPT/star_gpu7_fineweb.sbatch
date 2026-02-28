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

# =============================================================================
# Stage 1: Pretraining
# =============================================================================
echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py \
    --cache-gb 500 \
    --eval-every 200 \
    --eval-iters 12 \
    --batch-size 64 \
    --context 1024 \
    --grad-accum 4 \
    --num-workers 8 \
    --queue-size 25

echo "Pretraining finished: $(date)"

# Verify checkpoint exists before proceeding
if [ ! -f fineweb_gpt.ckpt ]; then
    echo "ERROR: fineweb_gpt.ckpt not found. Pretraining may have failed."
    exit 1
fi

# =============================================================================
# Stage 2: Chat Finetuning (SFT)
# =============================================================================
echo "========== STAGE 2: CHAT FINETUNING =========="
echo "Start: $(date)"

python finetune_chat.py \
    --ckpt fineweb_gpt.ckpt \
    --tok tokenizer.model \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 2e-5 \
    --eval-every 200 \
    --log-every 10

echo "Finetuning finished: $(date)"
echo "========== ALL DONE =========="
