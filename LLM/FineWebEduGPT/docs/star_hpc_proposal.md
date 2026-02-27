# Project Proposal: Scaling Laws for Educational Language Models

**Submitted to:** Hofstra University Star HPC Committee
**Principal Investigator:** Adam Naghavi, B.S. Data Science (Expected 2027)
**Department:** Computer Science / Data Science
**Date:** February 2026

---

## 1. Project Summary

This project investigates neural scaling laws by training a family of decoder-only transformer language models (125M, 355M, 760M, and 1.3B parameters) on FineWeb-Edu, a curated 1.3-trillion-token dataset of high-quality educational web text. The goal is to empirically characterize how model size, dataset size, and compute budget interact to determine downstream performance on educational and reasoning benchmarks -- replicating and extending findings from Hoffmann et al. (2022) and Kaplan et al. (2020) at an accessible scale.

All training code, model weights, and analysis notebooks will be open-sourced under the MIT license, providing a reproducible reference for the Hofstra research community and the broader academic ML ecosystem.

## 2. Scientific Motivation

### 2.1 Why Scaling Laws Matter

The "Chinchilla" scaling laws (Hoffmann et al., 2022) demonstrated that most large language models are significantly undertrained relative to their parameter count, and that optimal performance for a fixed compute budget requires jointly scaling model size and training data. However, these results were established on general web crawl data. It remains an open question whether the same relationships hold on domain-restricted, quality-filtered corpora -- particularly educational text, where information density is higher and stylistic variance is lower than raw web data.

### 2.2 FineWeb-Edu: A Curated Educational Corpus

FineWeb-Edu (Penedo et al., 2024) is a 1.3-trillion-token subset of FineWeb, filtered for educational quality using a classifier trained on annotations from Llama-3-70B-Instruct. Documents scoring 3+ on a 5-point educational value scale are retained. This filtering produces text that is substantially more information-dense than Common Crawl, making it an ideal testbed for studying whether scaling laws shift under improved data quality.

The dataset is openly available on HuggingFace and organized by Common Crawl snapshot (e.g., CC-MAIN-2025-26), enabling controlled experiments across data recency.

### 2.3 Research Questions

1. **Do Chinchilla-optimal compute allocations hold on curated educational text?** We hypothesize that higher data quality shifts the optimal model-size-to-tokens ratio, potentially favoring smaller models trained longer.
2. **How does model scale affect performance on educational benchmarks?** We will evaluate on ARC (Clark et al., 2018), HellaSwag (Zellers et al., 2019), and MMLU (Hendrycks et al., 2021) to measure reasoning and knowledge acquisition as a function of parameters.
3. **What is the compute-efficient frontier for educational LLMs?** By training models of varying sizes to different token counts, we can map the loss-vs-compute Pareto frontier specific to educational data.

## 3. Model Architecture

All models use a standard GPT-2-style decoder-only transformer with pre-norm (LayerNorm before attention/FFN), learned positional embeddings, GELU activations, and no bias terms in attention projections. This architecture is well-understood, easy to reproduce, and directly comparable to published scaling law results.

| Model | Parameters | Layers | Heads | d_model | Context | Vocab |
|-------|-----------|--------|-------|---------|---------|-------|
| EduGPT-125M | ~125M | 12 | 12 | 768 | 1024 | 32,000 |
| EduGPT-355M | ~355M | 20 | 16 | 1,024 | 1024 | 32,000 |
| EduGPT-760M | ~760M | 24 | 16 | 1,536 | 2048 | 32,000 |
| EduGPT-1.3B | ~1.3B | 24 | 32 | 2,048 | 2048 | 32,000 |

Tokenization uses a BPE tokenizer (SentencePiece) trained on a 100K-document sample of FineWeb-Edu.

## 4. Training Infrastructure and Methodology

### 4.1 Why Star HPC

The Star HPC cluster is uniquely suited to this project:

- **GPU capacity:** Training the 1.3B model to Chinchilla-optimal token counts (~26B tokens) requires approximately 2,400 H100 GPU-hours. The cluster's gpu5/gpu6 (4x H100 SXM each) and gpu7 (8x H100 SXM) provide the necessary multi-GPU parallelism.
- **Storage:** The rolling-cache data pipeline (see Section 4.3) requires 50-100 GB of temporary scratch space per node, well within the cluster's 63 TB parallel file system.
- **InfiniBand:** HDR200 InfiniBand enables efficient gradient synchronization via NCCL for Distributed Data Parallel training across multiple GPUs.
- **Software stack:** SLURM job scheduling, Apptainer containerization, and NGC image support align directly with the project's PyTorch + HuggingFace Datasets toolchain.

### 4.2 Distributed Training

The training script uses PyTorch's DistributedDataParallel (DDP) with NCCL backend. All models in this study fit in a single H100's 80GB HBM3, so DDP (data parallelism) is sufficient -- no model parallelism or FSDP is required. This keeps the implementation simple and reproducible.

Key DDP features implemented:
- Per-rank GPU binding via `LOCAL_RANK`
- Disjoint dataset sharding (each rank processes a unique document subset)
- All-reduce averaged evaluation loss across ranks
- Rank-0-only logging, checkpointing, and artifact saving
- Cosine learning rate schedule with linear warmup
- Mixed-precision training (FP16 via `torch.amp`) with gradient scaling
- `torch.compile` for kernel fusion on H100 architecture

### 4.3 Data Pipeline

FineWeb-Edu totals approximately 10 TB on disk. To avoid monopolizing cluster storage, the training script implements a **rolling cache** strategy:

1. Download N GB of dataset shards (parquet files) to local scratch
2. Load as a memory-mapped Arrow dataset for near-instant random access
3. Train on the cached shard batch with multi-threaded tokenization
4. Delete cached shards, download next batch, repeat

Progress state is persisted to disk, enabling seamless resume across SLURM job restarts. The cache size is configurable (e.g., `--cache-gb 50`), making the pipeline adaptable to whatever scratch allocation is available.

### 4.4 Estimated Compute Requirements

| Model | Tokens | Est. GPU-hours (H100) | Preferred Node |
|-------|--------|-----------------------|----------------|
| EduGPT-125M | 2.5B | ~50 | gpu3 or gpu4 (2x H100) |
| EduGPT-355M | 7.1B | ~200 | gpu5 or gpu6 (4x H100) |
| EduGPT-760M | 15.2B | ~800 | gpu5 or gpu6 (4x H100) |
| EduGPT-1.3B | 26B | ~2,400 | gpu7 (8x H100) |

**Total estimated compute: ~3,450 H100 GPU-hours** across all four model sizes.

Token counts are set at the Chinchilla-optimal ratio of ~20 tokens per parameter. Training will proceed smallest-to-largest, with each run validating the pipeline and establishing baselines before committing to larger allocations.

### 4.5 Storage Requirements

| Item | Size |
|------|------|
| Rolling data cache (temporary) | 50-100 GB |
| Model checkpoints (all sizes, periodic) | ~30 GB |
| Final model weights (all sizes) | ~8 GB |
| Tokenizer artifacts | < 10 MB |
| Evaluation logs and analysis | < 1 GB |
| **Total persistent storage** | **~40 GB** |

The bulk of the dataset is never stored persistently -- the rolling cache ensures only a small working set occupies disk at any time.

## 5. Evaluation Plan

### 5.1 Training Metrics
- Validation loss (cross-entropy) tracked every 500 steps
- Perplexity on held-out FineWeb-Edu validation split
- Training throughput (tokens/second) to verify hardware utilization

### 5.2 Downstream Benchmarks
Using the LM Evaluation Harness (Gao et al., 2021):
- **ARC-Easy / ARC-Challenge** -- science question answering
- **HellaSwag** -- commonsense reasoning
- **MMLU** -- multitask knowledge across 57 subjects
- **WinoGrande** -- coreference resolution

### 5.3 Scaling Analysis
- Log-log plots of validation loss vs. compute (FLOPs)
- Comparison of empirical scaling exponents against published Chinchilla coefficients
- Compute-optimal frontier analysis specific to educational data

## 6. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **1. Setup** | Week 1-2 | Account provisioning, Apptainer image build, tokenizer training, pipeline validation on gpu3/gpu4 |
| **2. Small-scale** | Week 3-4 | Train EduGPT-125M and EduGPT-355M, validate scaling trajectory |
| **3. Medium-scale** | Week 5-8 | Train EduGPT-760M, intermediate evaluation, checkpoint analysis |
| **4. Large-scale** | Week 9-14 | Train EduGPT-1.3B, full evaluation suite |
| **5. Analysis** | Week 15-16 | Scaling law fitting, benchmark evaluation, paper drafting |

## 7. Deliverables

1. **Open-source training code** -- full DDP-enabled GPT trainer with rolling cache data pipeline (MIT license, hosted on GitHub)
2. **Trained model weights** -- all four model sizes, available on HuggingFace Hub
3. **Scaling law analysis** -- empirical characterization of the loss-vs-compute relationship on educational data, with comparison to published results on general web data
4. **Technical report / paper** -- suitable for submission to a workshop or conference (e.g., NeurIPS Efficient NLP, EMNLP)
5. **Reproducibility package** -- SLURM job scripts, Apptainer definition files, and evaluation notebooks for full replication on Star HPC

## 8. Broader Impact

This project directly serves the Star HPC's mission of exposing students to HPC technology and fostering research collaboration:

- **Educational value:** The entire pipeline -- from data processing to distributed training to scaling analysis -- is a practical curriculum in modern ML systems engineering.
- **Community resource:** Open-sourced artifacts (code, weights, analysis) provide a starting point for other Hofstra students and faculty working on NLP, education technology, or computational linguistics.
- **Reproducibility:** The rolling-cache data pipeline and containerized training environment make the project fully reproducible on Star HPC without requiring terabytes of persistent storage.
- **Benchmarking:** The project establishes baseline LLM training benchmarks on Star HPC hardware, useful for future resource planning and allocation decisions.

## 9. References

- Clark, P., et al. (2018). Think you have Solved Question Answering? Try ARC. *arXiv:1803.05457*.
- Gao, L., et al. (2021). A framework for few-shot language model evaluation. *Zenodo*.
- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR*.
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *arXiv:2203.15556*.
- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
- Penedo, G., et al. (2024). FineWeb-Edu: The Finest Collection of Educational Data the Web Has to Offer. *arXiv*.
- Zellers, R., et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence? *ACL*.

---

*Code repository: [github.com/AdamNaghs/models](https://github.com/AdamNaghs/models)*
