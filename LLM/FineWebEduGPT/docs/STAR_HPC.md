# Star HPC Runbook

This runbook is the operator guide for training FineWebEduGPT on Star with manually staged FineWeb-Edu sample chunks.

## Paths

Shared storage root:

```text
/fs1/proj/educational_web_data
```

Default sample workflow paths:

```text
/fs1/proj/educational_web_data/logs
/fs1/proj/educational_web_data/runs/1.3b
/fs1/proj/educational_web_data/runs/1.3b-smoke
/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source
/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source
```

## One-Time Setup

From the project directory on the Star login node:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
python3 -m venv llmvenv
source llmvenv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
mkdir -p /fs1/proj/educational_web_data/logs
```

## Workflow

Choose one sample config for the run:
- `sample-10BT` for smoke validation
- `sample-100BT` for the real `1.3b` run
- do not combine nested samples

### Step 1: Stage the next chunk

Smoke run:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
source llmvenv/bin/activate
python download_fineweb_snapshot.py --config sample-10BT --max-gb 500
```

Real `1.3b` run:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
source llmvenv/bin/activate
python download_fineweb_snapshot.py --config sample-100BT --max-gb 500
```

Optional checks:

```bash
find /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source -name '*.parquet' | head
cat /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/.download_state.json
cat /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source/_chunk_manifest.json
```

### Step 2: Submit the smoke run

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source
sbatch --qos=long2x \
  --export=ALL,OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b-smoke,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=sample-10BT,BATCH_SIZE=1,GRAD_ACCUM=32,NO_COMPILE=1,TRAIN_STEPS=20,EVAL_EVERY=10,EVAL_ITERS=2,CKPT_EVERY=20 \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-smoke-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-smoke-%j.err \
  star_fineweb_1_3b.sbatch
```

Smoke acceptance:
- dataset indexing completes
- first backward pass succeeds
- no CUDA OOM appears in stderr
- at least one eval or checkpoint is written

### Step 3: Submit the real `1.3b` run

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
sbatch star_fineweb_1_3b.sbatch
```

Default `1.3b` Star settings:
- `BATCH_SIZE=1`
- `GRAD_ACCUM=128`
- `NO_COMPILE=1`
- `CONFIG=sample-100BT`
- `OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b`
- `QOS=long2x`
- offline staged training streams parquet directly and does not build a second large Arrow cache on disk
- offline staged training skips the expensive step-0 validation pass by default
- staged parquet row groups are balanced across ranks by row count before training starts

### Step 4: Check job status

After submission, note the job ID from `sbatch`.

```bash
squeue -j <jobid>
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed
scontrol show job <jobid>
```

Live resource checks:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
sstat -j <jobid>.batch --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveVMSize
```

### Step 5: View the logs

```bash
tail -f /fs1/proj/educational_web_data/logs/fineweb-1-3b-<jobid>.out
tail -f /fs1/proj/educational_web_data/logs/fineweb-1-3b-<jobid>.err
```

Quick snapshots:

```bash
tail -n 50 /fs1/proj/educational_web_data/logs/fineweb-1-3b-<jobid>.out
tail -n 50 /fs1/proj/educational_web_data/logs/fineweb-1-3b-<jobid>.err
```

Expected startup behavior for offline staged runs:
- stdout should report direct parquet streaming and row-balanced rank assignment
- there is no initial `step 0 | val ...` line before the first training step
- the first visible training progress should be a later step/eval log, not a long silent pre-eval phase
- if a rank cannot get batches, the job now fails with a direct staged-parquet batch timeout instead of waiting for a multi-hour NCCL watchdog timeout

### Step 6: Wait for the stop condition

The job should stop by itself when either:
- it reaches the preset `train_steps`
- it completes one full pass over the currently staged sample chunk

When it finishes the current chunk early, stdout should include:

```text
data: completed one full pass over the staged local chunk
Chunk training finished. Run download_fineweb_snapshot.py again to stage the next chunk for this sample, then resubmit with --resume.
```

### Step 7: Repeat for the next chunk

If pretraining is not finished yet:

```bash
python download_fineweb_snapshot.py --config sample-100BT --max-gb 500
sbatch star_fineweb_1_3b.sbatch
```

Keep using the same `OUT_DIR` so pretraining resumes from the latest checkpoint.

## After Pretraining

Evaluate the pretrained checkpoint before post-training:

```bash
source llmvenv/bin/activate
OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b
TOK=$OUT_DIR/tokenizer.model
CKPT=$OUT_DIR/fineweb_gpt.ckpt

python eval/eval_lm.py --ckpt "$CKPT" --tok "$TOK" --dataset wikitext_valid --limit 128
python eval/eval_mcq.py --ckpt "$CKPT" --tok "$TOK" --bench hellaswag
python eval/eval_mcq.py --ckpt "$CKPT" --tok "$TOK" --bench arc_challenge
```
