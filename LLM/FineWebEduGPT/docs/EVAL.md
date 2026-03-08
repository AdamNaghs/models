# FineWebEduGPT Evaluation Guide

This repo includes a lightweight evaluation stack for three jobs:

1. **Language-model sanity checks** (perplexity on held-out text)
2. **Multiple-choice benchmark scoring** (HellaSwag, PIQA, Winogrande, ARC)
3. **Chat regression testing** (fixed prompt suite)
4. **Contamination scanning** (exact + n-gram overlap audit)

The goal is fast signal without forcing a conversion to Hugging Face model format.

---

## Files

```text
FineWebEduGPT/
├── eval/
│   ├── model_adapter.py        # checkpoint wrapper: scoring + generation + perplexity
│   ├── benchmark_loaders.py    # HF dataset loaders -> normalized benchmark rows
│   ├── eval_lm.py              # held-out perplexity / avg NLL
│   ├── eval_mcq.py             # multiple-choice benchmark runner
│   ├── eval_chat.py            # fixed prompt chat regression runner
│   ├── contamination_scan.py   # contamination audit over local files / FineWeb samples
│   ├── metrics.py              # simple aggregate metrics
│   └── utils.py                # JSONL, normalization, n-grams, helpers
├── eval_data/
│   └── chat_eval_prompts.jsonl # curated prompt suite for chat checks
└── run_eval.sh                 # convenience wrapper
```

---

## Install

```bash
pip install -r requirements.txt
python smoke_test.py
```

The eval stack uses the same checkpoint format already produced by this repo.

---

## 1) Held-out LM sanity check

Run a quick perplexity pass on an external text set:

```bash
OUT_DIR=runs/350m
python eval/eval_lm.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --dataset wikitext_valid
```

Supported datasets:

- `wikitext_valid`
- `wikitext_test`
- `lambada`

Output: JSON summary under `eval_results/<checkpoint-name>/`.

Why this matters:
- if perplexity is terrible, benchmark scores downstream will not save you
- it is the fastest way to detect a model that never really learned language

---

## 2) Multiple-choice benchmarks

Run one benchmark at a time:

```bash
OUT_DIR=runs/350m
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench hellaswag
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench piqa
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench winogrande
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench arc_challenge
```

Supported benchmarks:

- `hellaswag`
- `piqa`
- `winogrande`
- `arc_easy`
- `arc_challenge`

Notes:
- `eval_mcq.py` uses teacher-forced log-likelihood over each answer choice
- default selection metric is **length-normalized logprob**, which is more fair when answer lengths vary
- use `--metric raw` if you want raw total logprob instead

Example with a sample limit:

```bash
python eval/eval_mcq.py --ckpt runs/350m/fineweb_gpt.ckpt --bench hellaswag --limit 200
```

---

## 3) Chat regression suite

Run the fixed prompt pack:

```bash
OUT_DIR=runs/350m
python eval/eval_chat.py --ckpt "$OUT_DIR/fineweb_gpt_chat.ckpt" --prompts eval_data/chat_eval_prompts.jsonl
```

What it covers:
- factual QA
- summarization
- reasoning
- structured output
- safety/refusal
- coding-lite prompts
- multi-turn memory
- hallucination / uncertainty handling

Outputs:
- JSONL file of prompt → response pairs
- Markdown report for human review

This is not an automatic leaderboard score. It is a **regression suite** so you can compare runs and spot behavioral drift.

---

## 4) Full quick-run wrapper

Run the default bundle:

```bash
OUT_DIR=runs/350m
bash run_eval.sh "$OUT_DIR/fineweb_gpt_chat.ckpt"
```

`run_eval.sh` automatically derives the base checkpoint as `$(dirname "$CHAT_CKPT")/fineweb_gpt.ckpt` unless you override `BASE_CKPT`.

Environment knobs:

```bash
RESULTS_DIR=eval_results LIMIT=200 bash run_eval.sh runs/350m/fineweb_gpt_chat.ckpt
BASE_CKPT=runs/350m/fineweb_gpt.ckpt bash run_eval.sh runs/350m/fineweb_gpt_chat.ckpt
```

What it runs by default:
- `eval_lm.py` on `wikitext_valid`
- `eval_mcq.py` on HellaSwag / PIQA / Winogrande / ARC-Challenge
- `eval_chat.py` on `eval_data/chat_eval_prompts.jsonl`

---

## 5) Contamination scanning

The contamination scanner supports two source types:

1. **Local files/directories** you want to scan
2. **Sampled FineWeb-Edu docs** streamed from Hugging Face

### Scan benchmark items against sampled FineWeb docs

```bash
python eval/contamination_scan.py \
  --bench hellaswag \
  --bench piqa \
  --fineweb-config CC-MAIN-2025-26 \
  --fineweb-sample-docs 5000 \
  --ckpt-label fineweb-cc-main-2025-26
```

### Scan local text directories

```bash
python eval/contamination_scan.py \
  --bench hellaswag \
  --source-path data/heldout_text \
  --source-path notes/
```

### Scan an existing manifest

```bash
python eval/contamination_scan.py \
  --manifest eval_results/fineweb-cc-main-2025-26/benchmark_manifest.jsonl \
  --source-path /path/to/text-corpus
```

Outputs:
- `benchmark_manifest.jsonl`
- `contamination_findings.jsonl`
- `contamination_summary.json`

Status meanings:
- `clean`: no strong overlap found in the scanned source set
- `suspected`: exact prompt overlap or high n-gram overlap
- `contaminated`: exact prompt + gold-answer overlap found

Important: if you only scan a sample of FineWeb, this is a **sampled contamination audit**, not a proof of cleanliness.

---

## Recommended order of operations

1. Run `eval_lm.py`
2. Run `eval_mcq.py` on the four core benchmarks
3. Run `eval_chat.py`
4. Run `contamination_scan.py`
5. Report both:
   - raw benchmark scores
   - decontaminated benchmark scores after excluding suspicious items

That order matters. If the base model is weak, contamination analysis won’t rescue it.

---

## Practical interpretation

### If perplexity is bad
The model probably did not learn enough language. Fix training budget, data coverage, or tokenizer quality before obsessing over chat polish.

### If MCQ scores are strong but chat is weak
The model may have decent base knowledge but weak SFT quality, weak prompting, or benchmark contamination.

### If chat is decent but MCQ is weak
That usually means the SFT helped surface behavior, but the base model lacks depth.

### If contamination flags are high
Report benchmark scores two ways:
- raw
- filtered / decontaminated

Do not trust raw leaderboard-style numbers if the benchmark text appears in training data.
