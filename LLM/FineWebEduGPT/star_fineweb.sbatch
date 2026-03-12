#!/bin/bash
#SBATCH --job-name=fineweb-train
#SBATCH --output=/fs1/proj/educational_web_data/logs/fineweb-train-%j.out
#SBATCH --error=/fs1/proj/educational_web_data/logs/fineweb-train-%j.err
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

PRESET="${PRESET:-350m}"
STORAGE_ROOT="${FINEWEB_STORAGE_ROOT:-/fs1/proj/educational_web_data}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/logs}"
OUT_DIR="${OUT_DIR:-$STORAGE_ROOT/runs/$PRESET}"
CONFIG="sample-100BT"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$STORAGE_ROOT/dataset/fineweb-edu/$CONFIG/source}"
mkdir -p "$LOG_DIR" "$OUT_DIR"
mkdir -p "$LOCAL_DATA_DIR"
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
echo "Config: $CONFIG"
echo "Local staged data dir: $LOCAL_DATA_DIR"

# =============================================================================
# Stage 1: Pretraining
# =============================================================================
echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

TRAIN_ARGS=(
    --preset "$PRESET"
    --config "$CONFIG"
    --offline
    --out-dir "$OUT_DIR"
    --num-workers 8
    --queue-size 25
    --stop-after-one-epoch
    --local-data-dir "$LOCAL_DATA_DIR"
)

if [ -f "$OUT_DIR/fineweb_gpt.ckpt" ]; then
    echo "Resuming pretraining from $OUT_DIR/fineweb_gpt.ckpt"
    TRAIN_ARGS+=(--resume "$OUT_DIR/fineweb_gpt.ckpt")
fi

torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py "${TRAIN_ARGS[@]}"

echo "Pretraining finished: $(date)"

# Verify checkpoint exists before proceeding
if [ ! -f "$OUT_DIR/fineweb_gpt.ckpt" ]; then
    echo "ERROR: $OUT_DIR/fineweb_gpt.ckpt not found. Pretraining may have failed."
    exit 1
fi

echo "Chunk training finished."
echo "Run download_fineweb_snapshot.py again to stage the next chunk for this sample, then resubmit this sbatch."
