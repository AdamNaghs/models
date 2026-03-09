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

PRESET="${PRESET:-350m}"
STORAGE_ROOT="${FINEWEB_STORAGE_ROOT:-/fs1/proj/educational_web_data}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
OUT_DIR="${OUT_DIR:-$STORAGE_ROOT/runs/$PRESET}"
CONFIG="${CONFIG:-CC-MAIN-2025-26}"
CACHE_DIR="${CACHE_DIR:-$STORAGE_ROOT/dataset/fineweb-edu/$CONFIG/$PRESET}"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$CACHE_DIR"
cd "$SLURM_SUBMIT_DIR"

# Activate your environment (edit if needed)
source llmvenv/bin/activate

pip install -r requirements.txt

# Helpful runtime diagnostics
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "Artifacts will be written to: $OUT_DIR"
echo "Preset: $PRESET"
echo "Storage root: $STORAGE_ROOT"
echo "Log directory: $LOG_DIR"
echo "Dataset cache will be written to: $CACHE_DIR"

# =============================================================================
# Stage 1: Pretraining
# =============================================================================
echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py \
    --preset "$PRESET" \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --cache-gb 500 \
    --num-workers 8 \
    --queue-size 25

echo "Pretraining finished: $(date)"

# Verify checkpoint exists before proceeding
if [ ! -f "$OUT_DIR/fineweb_gpt.ckpt" ]; then
    echo "ERROR: $OUT_DIR/fineweb_gpt.ckpt not found. Pretraining may have failed."
    exit 1
fi

# =============================================================================
# Stage 2: Chat Finetuning (SFT)
# =============================================================================
echo "========== STAGE 2: CHAT FINETUNING =========="
echo "Start: $(date)"

python finetune_chat.py \
    --ckpt "$OUT_DIR/fineweb_gpt.ckpt" \
    --output "$OUT_DIR/fineweb_gpt_chat.ckpt" \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 2e-5 \
    --eval-every 200 \
    --log-every 10

echo "Finetuning finished: $(date)"
echo "========== ALL DONE =========="
