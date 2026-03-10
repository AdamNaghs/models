#!/bin/bash
#SBATCH --job-name=fineweb-1-3b
#SBATCH --output=fineweb-1-3b-%j.out
#SBATCH --error=fineweb-1-3b-%j.err
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

PRESET="1.3b"
STORAGE_ROOT="${FINEWEB_STORAGE_ROOT:-/fs1/proj/educational_web_data}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
OUT_DIR="${OUT_DIR:-$STORAGE_ROOT/runs/1.3b}"
CONFIGS="${CONFIGS:-CC-MAIN-2025-05:CC-MAIN-2025-08:CC-MAIN-2025-13:CC-MAIN-2025-18:CC-MAIN-2025-21:CC-MAIN-2025-26}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-128}"
NO_COMPILE="${NO_COMPILE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
EVAL_ITERS="${EVAL_ITERS:-}"
CKPT_EVERY="${CKPT_EVERY:-}"
LOCAL_DATA_DIRS="${LOCAL_DATA_DIRS:-}"

IFS=':' read -r -a CONFIG_LIST <<< "$CONFIGS"
LOCAL_DIR_ARGS=()
if [ -n "$LOCAL_DATA_DIRS" ]; then
    IFS=':' read -r -a LOCAL_DIR_LIST <<< "$LOCAL_DATA_DIRS"
else
    LOCAL_DIR_LIST=()
    for cfg in "${CONFIG_LIST[@]}"; do
        LOCAL_DIR_LIST+=("$STORAGE_ROOT/dataset/fineweb-edu/$cfg/source")
    done
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"
for local_dir in "${LOCAL_DIR_LIST[@]}"; do
    mkdir -p "$local_dir"
    LOCAL_DIR_ARGS+=(--local-data-dir "$local_dir")
done
cd "$SLURM_SUBMIT_DIR"

# Activate your environment (edit if needed)
source llmvenv/bin/activate

pip install -r requirements.txt

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "Preset: $PRESET"
echo "Configs: $CONFIGS"
echo "Storage root: $STORAGE_ROOT"
echo "Log directory: $LOG_DIR"
echo "Artifacts will be written to: $OUT_DIR"
echo "Local staged data dirs: ${LOCAL_DIR_LIST[*]}"
echo "Micro-batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM"
echo "torch.compile disabled: $NO_COMPILE"

echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

TRAIN_ARGS=(
    --preset "$PRESET"
    --config "${CONFIG_LIST[0]}"
    --offline
    --out-dir "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --num-workers 8
    --queue-size 25
    --stop-after-one-epoch
)
TRAIN_ARGS+=("${LOCAL_DIR_ARGS[@]}")

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
echo "Run download_fineweb_snapshot.py again to stage more configs or additional chunks, then resubmit this sbatch."
