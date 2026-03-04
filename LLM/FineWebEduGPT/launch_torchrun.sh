#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./launch_torchrun.sh 350m 8
#   ./launch_torchrun.sh 125m 8
#   ./launch_torchrun.sh 1.3b 8

PRESET="${1:-350m}"
NPROC="${2:-8}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  train_fineweb_gpt.py \
  --preset "$PRESET"
