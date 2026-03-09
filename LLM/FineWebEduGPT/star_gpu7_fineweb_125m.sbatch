#!/bin/bash
#SBATCH --job-name=fineweb-125m
#SBATCH --output=fineweb-125m-%j.out
#SBATCH --error=fineweb-125m-%j.err
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

PRESET="125m"
STORAGE_ROOT="${FINEWEB_STORAGE_ROOT:-/fs1/proj/educational_web_data}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
OUT_DIR="${OUT_DIR:-$STORAGE_ROOT/runs/125m}"
CONFIG="${CONFIG:-CC-MAIN-2025-26}"
NPROC="${NPROC:-8}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$STORAGE_ROOT/dataset/fineweb-edu/$CONFIG/source}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$LOCAL_DATA_DIR"
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

echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

TRAIN_ARGS=(
    --preset "$PRESET"
    --config "$CONFIG"
    --offline
    --local-data-dir "$LOCAL_DATA_DIR"
    --out-dir "$OUT_DIR"
    --num-workers 8
    --queue-size 25
    --stop-after-one-epoch
)

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
echo "Run download_fineweb_snapshot.py again to stage the next chunk, then resubmit this sbatch."
