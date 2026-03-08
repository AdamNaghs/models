#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./launch_torchrun.sh 350m 8
#   ./launch_torchrun.sh 125m 8 runs/125m
#   ./launch_torchrun.sh 1.3b 8 /scratch/$USER/fineweb-1.3b

PRESET="${1:-350m}"
NPROC="${2:-8}"
OUT_DIR="${3:-runs/$PRESET}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

echo "preset=$PRESET nproc=$NPROC out_dir=$OUT_DIR"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train_fineweb_gpt.py \
  --preset "$PRESET" \
  --out-dir "$OUT_DIR"
