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
CONFIGS="${CONFIGS:-CC-MAIN-2025-26}"
LOCAL_DATA_DIRS="${LOCAL_DATA_DIRS:-}"
IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
LOCAL_DIR_ARGS=()
if [ -n "$LOCAL_DATA_DIRS" ]; then
    IFS=',' read -r -a LOCAL_DIR_LIST <<< "$LOCAL_DATA_DIRS"
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

# Helpful runtime diagnostics
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "Artifacts will be written to: $OUT_DIR"
echo "Preset: $PRESET"
echo "Storage root: $STORAGE_ROOT"
echo "Log directory: $LOG_DIR"
echo "Configs: $CONFIGS"
echo "Local staged data dirs: ${LOCAL_DIR_LIST[*]}"

# =============================================================================
# Stage 1: Pretraining
# =============================================================================
echo "========== STAGE 1: PRETRAINING =========="
echo "Start: $(date)"

TRAIN_ARGS=(
    --preset "$PRESET"
    --config "${CONFIG_LIST[0]}"
    --offline
    --out-dir "$OUT_DIR"
    --num-workers 8
    --queue-size 25
    --stop-after-one-epoch
)
TRAIN_ARGS+=("${LOCAL_DIR_ARGS[@]}")

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
echo "Run download_fineweb_snapshot.py again to stage more configs or additional chunks, then resubmit this sbatch."
