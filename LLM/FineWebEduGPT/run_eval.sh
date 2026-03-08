#!/usr/bin/env bash
set -euo pipefail

CHAT_CKPT="${1:-runs/350m/fineweb_gpt_chat.ckpt}"
TOK="${2:-}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
LIMIT="${LIMIT:-}"
BASE_CKPT="${BASE_CKPT:-$(dirname "$CHAT_CKPT")/fineweb_gpt.ckpt}"

BASE_ARGS=(--ckpt "$BASE_CKPT" --results-dir "$RESULTS_DIR")
CHAT_ARGS=(--ckpt "$CHAT_CKPT" --results-dir "$RESULTS_DIR")
if [[ -n "$TOK" ]]; then
  BASE_ARGS+=(--tok "$TOK")
  CHAT_ARGS+=(--tok "$TOK")
fi
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
else
  LIMIT_ARGS=()
fi

python eval/eval_lm.py "${BASE_ARGS[@]}" --dataset wikitext_valid "${LIMIT_ARGS[@]}"
python eval/eval_mcq.py "${BASE_ARGS[@]}" --bench hellaswag "${LIMIT_ARGS[@]}"
python eval/eval_mcq.py "${BASE_ARGS[@]}" --bench piqa "${LIMIT_ARGS[@]}"
python eval/eval_mcq.py "${BASE_ARGS[@]}" --bench winogrande "${LIMIT_ARGS[@]}"
python eval/eval_mcq.py "${BASE_ARGS[@]}" --bench arc_challenge "${LIMIT_ARGS[@]}"
python eval/eval_chat.py "${CHAT_ARGS[@]}" --prompts eval_data/chat_eval_prompts.jsonl

echo "Base checkpoint: $BASE_CKPT"
echo "Chat checkpoint: $CHAT_CKPT"
echo "Eval complete. Results under: $RESULTS_DIR"
