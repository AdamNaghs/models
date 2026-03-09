#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./launch_torchrun.sh 350m 8
#   ./launch_torchrun.sh 125m 8 /fs1/proj/educational_web_data/runs/125m
#   ./launch_torchrun.sh 1.3b 8 /fs1/proj/educational_web_data/runs/1.3b

PRESET="${1:-350m}"
NPROC="${2:-8}"
STORAGE_ROOT="${FINEWEB_STORAGE_ROOT:-/fs1/proj/educational_web_data}"
OUT_DIR="${3:-$STORAGE_ROOT/runs/$PRESET}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

echo "preset=$PRESET nproc=$NPROC storage_root=$STORAGE_ROOT out_dir=$OUT_DIR"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train_fineweb_gpt.py \
  --preset "$PRESET" \
  --out-dir "$OUT_DIR"
