#!/bin/bash
#SBATCH --job-name=fineweb-1-3b
#SBATCH --output=/fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.out
#SBATCH --error=/fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --qos=long2x

# Optional (uncomment if your cluster requires them)
##SBATCH --partition=gpu
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@hofstra.edu

set -euo pipefail

PRESET="1.3b"
STORAGE_ROOT="${FINEWEB_STORAGE_ROOT:-/fs1/proj/educational_web_data}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
OUT_DIR="${OUT_DIR:-$STORAGE_ROOT/runs/1.3b}"
CONFIG="sample-100BT"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-128}"
NO_COMPILE="${NO_COMPILE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
EVAL_ITERS="${EVAL_ITERS:-}"
CKPT_EVERY="${CKPT_EVERY:-}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$STORAGE_ROOT/dataset/fineweb-edu/$CONFIG/source}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
mkdir -p "$LOCAL_DATA_DIR"
cd "$SLURM_SUBMIT_DIR"

# Activate your environment (edit if needed)
source llmvenv/bin/activate

pip install -r requirements.txt

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "Preset: $PRESET"
echo "Config: $CONFIG"
echo "Storage root: $STORAGE_ROOT"
echo "Log directory: $LOG_DIR"
echo "Artifacts will be written to: $OUT_DIR"
echo "Local staged data dir: $LOCAL_DATA_DIR"
echo "Micro-batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "torch.compile disabled: $NO_COMPILE"

echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

TRAIN_ARGS=(
    --preset "$PRESET"
    --config "$CONFIG"
    --offline
    --out-dir "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --num-workers 8
    --queue-size 25
    --stop-after-one-epoch
    --local-data-dir "$LOCAL_DATA_DIR"
)

if [ "$NO_COMPILE" = "1" ]; then
    TRAIN_ARGS+=(--no-compile)
fi

if [ -n "$TRAIN_STEPS" ]; then
    TRAIN_ARGS+=(--train-steps "$TRAIN_STEPS")
fi

if [ -n "$EVAL_EVERY" ]; then
    TRAIN_ARGS+=(--eval-every "$EVAL_EVERY")
fi

if [ -n "$EVAL_ITERS" ]; then
    TRAIN_ARGS+=(--eval-iters "$EVAL_ITERS")
fi

if [ -n "$CKPT_EVERY" ]; then
    TRAIN_ARGS+=(--ckpt-every "$CKPT_EVERY")
fi

if [ -f "$OUT_DIR/fineweb_gpt.ckpt" ]; then
    echo "Resuming pretraining from $OUT_DIR/fineweb_gpt.ckpt"
    TRAIN_ARGS+=(--resume "$OUT_DIR/fineweb_gpt.ckpt")
fi

torchrun --standalone --nproc_per_node="$NPROC" train_fineweb_gpt.py "${TRAIN_ARGS[@]}"

echo "Pretraining finished: $(date)"

if [ ! -f "$OUT_DIR/fineweb_gpt.ckpt" ]; then
    echo "ERROR: $OUT_DIR/fineweb_gpt.ckpt not found. Pretraining may have failed."
    exit 1
fi

echo "Chunk training finished."
echo "Run download_fineweb_snapshot.py again to stage the next chunk for this sample, then resubmit this sbatch."
