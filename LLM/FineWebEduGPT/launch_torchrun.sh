#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./launch_torchrun.sh h100 8
#   ./launch_torchrun.sh a100 4

PRESET="${1:-h100}"
NPROC="${2:-8}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train_fineweb_gpt.py \
  --preset "$PRESET"
