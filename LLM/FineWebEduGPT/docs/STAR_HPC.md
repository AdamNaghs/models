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

### Step 1: Stage the next chunk

Run this on the login node:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
source llmvenv/bin/activate
python download_fineweb_snapshot.py \
  --config CC-MAIN-2025-26 \
  --max-gb 500
```

What it does:
- clears the previous staged chunk under `.../source`
- downloads the next shard window from FineWeb-Edu
- writes `_chunk_manifest.json` in the source directory
- advances `.download_state.json` so the next run downloads the following chunk

Optional checks:

```bash
find /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source -name '*.parquet' | head
cat /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/.download_state.json
```

### Step 2: Submit the training job

For the `125m` preset:

```bash
cd ~/edu_web_data/models/LLM/FineWebEduGPT
sbatch --qos=long2x \
  -o /fs1/proj/educational_web_data/logs/fineweb-125m-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-125m-%j.err \
  star_gpu7_fineweb_125m.sbatch
```

What the sbatch does:
- uses `--offline --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/<config>/source`
- resumes from `/fs1/proj/educational_web_data/runs/<preset>/fineweb_gpt.ckpt` if it exists
- trains only on the currently staged chunk
- stops after one full pass over that chunk

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

When it finishes a chunk early, the output should include a message like:

```text
data: completed one full pass over the staged local chunk
Chunk training finished. Run download_fineweb_snapshot.py again to stage the next chunk, then resubmit with --resume.
```

### Step 6: Repeat

If pretraining is not finished yet:

```bash
python download_fineweb_snapshot.py --config CC-MAIN-2025-26 --max-gb 500
sbatch --qos=long2x \
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
