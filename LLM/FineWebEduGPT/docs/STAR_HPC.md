# Star HPC Runbook

This runbook is the operator guide for training FineWebEduGPT on Star with the manual staged-chunk workflow.

## Paths

Shared storage root:

```text
/fs1/proj/educational_web_data
```

Key directories for the `125m` preset:

```text
/fs1/proj/educational_web_data/logs
/fs1/proj/educational_web_data/runs/125m
/fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source
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

## Training Pipeline

Run the following loop until pretraining reaches its final step target.

### Step 1: Stage the next config set

Run this on the login node:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
source llmvenv/bin/activate
python download_fineweb_snapshot.py \
  --config CC-MAIN-2025-21 \
  --config CC-MAIN-2025-26 \
  --max-gb 500
```

What it does:
- stages one `source/` directory per config
- downloads the next shard window from each requested FineWeb-Edu config
- writes `_chunk_manifest.json` in each source directory
- advances `.download_state.json` so the next run downloads the following chunk for each config

Optional checks:

```bash
find /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-21/source -name '*.parquet' | head
find /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source -name '*.parquet' | head
cat /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-21/.download_state.json
cat /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/.download_state.json
```

### Step 2: Submit the training job

For the `125m` preset:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-21/source:/fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source
sbatch --qos=long2x --export=ALL,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=CC-MAIN-2025-21:CC-MAIN-2025-26,BATCH_SIZE=8,GRAD_ACCUM=16 \
  -o /fs1/proj/educational_web_data/logs/fineweb-125m-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-125m-%j.err \
  star_gpu7_fineweb_125m.sbatch
```

What the sbatch does:
- uses `--offline` plus one `--local-data-dir` per staged config
- resumes from `/fs1/proj/educational_web_data/runs/<preset>/fineweb_gpt.ckpt` if it exists
- trains on the currently staged config set
- stops after one full pass over that chunk
- uses H100-safe `125m` settings: `BATCH_SIZE=8`, `GRAD_ACCUM=16`, and `--no-compile`

### Step 3: Check job status

After submission, note the job ID from `sbatch`.

Example:

```bash
sbatch ...
Submitted batch job 16348
```

Check whether it is queued or running:

```bash
squeue -j 16348
```

Check accounting status:

```bash
sacct -j 16348 --format=JobID,JobName,State,ExitCode,Elapsed
```

If the job is still active, inspect scheduler metadata:

```bash
scontrol show job 16348
```

### Step 4: View the logs

Watch stdout:

```bash
tail -f /fs1/proj/educational_web_data/logs/fineweb-125m-16348.out
```

Watch stderr:

```bash
tail -f /fs1/proj/educational_web_data/logs/fineweb-125m-16348.err
```

Quick snapshots instead of follow mode:

```bash
tail -n 50 /fs1/proj/educational_web_data/logs/fineweb-125m-16348.out
tail -n 50 /fs1/proj/educational_web_data/logs/fineweb-125m-16348.err
```

### Step 5: Wait for the stop condition

The job should stop by itself when either:
- it reaches the preset’s total `train_steps`
- it completes one full pass over the currently staged chunk

When it finishes the staged config set early, the output should include a message like:

```text
data: completed one full pass over the staged local chunk
Chunk training finished. Run download_fineweb_snapshot.py again to stage the next chunk, then resubmit with --resume.
```

### Step 6: Repeat

If pretraining is not finished yet:

```bash
python download_fineweb_snapshot.py --config CC-MAIN-2025-21 --config CC-MAIN-2025-26 --max-gb 500
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-21/source:/fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source
sbatch --qos=long2x --export=ALL,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=CC-MAIN-2025-21:CC-MAIN-2025-26,BATCH_SIZE=8,GRAD_ACCUM=16 \
  -o /fs1/proj/educational_web_data/logs/fineweb-125m-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-125m-%j.err \
  star_gpu7_fineweb_125m.sbatch
```

Keep using the same `OUT_DIR` so the checkpoint resumes from the previous chunk.

## Useful Commands

Cancel a running job:

```bash
scancel 16348
```

Check the latest pretraining checkpoint:

```bash
ls -lh /fs1/proj/educational_web_data/runs/125m/fineweb_gpt.ckpt
```

Check that the tokenizer exists:

```bash
ls -lh /fs1/proj/educational_web_data/runs/125m/tokenizer.model
```

Show the staged chunk manifest:

```bash
cat /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source/_chunk_manifest.json
```

## End Condition

You are done with pretraining when the job reaches the final training step target and no longer asks for another chunk.

Chat finetuning is intentionally not started automatically in the Star sbatch files. Run it separately after pretraining is complete.
