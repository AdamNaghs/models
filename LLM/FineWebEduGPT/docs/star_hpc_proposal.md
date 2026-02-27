# Project Proposal: Training Language Models on Educational Web Data

**Submitted by:** Adam Naghavi, B.S. Data Science (Expected 2027)
**Department:** Computer Science / Data Science
**Date:** February 2026

---

## Summary

This project explores training small-to-medium scale transformer language models on FineWeb-Edu, a large open-source dataset of educational web text curated by HuggingFace. The focus is on understanding how model size, training data volume, and compute budget affect language model performance -- particularly on educational and reasoning tasks.

The work involves building and training GPT-style models of varying sizes using PyTorch with distributed data parallel training across multiple GPUs. Models will be evaluated on standard NLP benchmarks to measure knowledge acquisition and reasoning ability as a function of scale.

## Motivation

Large language models are increasingly central to both industry and academic research, but most published training runs happen behind closed doors at large labs. Training models from scratch on curated data provides hands-on experience with the full ML pipeline -- data processing, distributed systems, optimization, and evaluation -- and produces artifacts that can be shared with the Hofstra research community.

FineWeb-Edu is a publicly available dataset filtered for educational quality, making it a natural fit for a university research context. Training on it allows the study of how data quality affects model learning without requiring proprietary data sources.

## Scope

- Train decoder-only transformer models ranging from ~125M to ~1.3B parameters
- Use PyTorch DDP (Distributed Data Parallel) for multi-GPU training
- Evaluate trained models on standard benchmarks (ARC, HellaSwag, MMLU)
- Open-source all code, weights, and evaluation results

## Resource Requirements

- **Compute:** Multi-GPU nodes (H100 or A100) for distributed training runs
- **Storage:** Minimal persistent footprint (~40 GB for checkpoints and artifacts). The training pipeline uses a rolling cache that downloads, trains on, and deletes data in chunks, avoiding large persistent storage needs.
- **Software:** PyTorch, HuggingFace Datasets, SentencePiece, SLURM. Compatible with Apptainer containerization.

## Deliverables

- Trained model weights (publicly released)
- Open-source training code and reproduction scripts
- Evaluation results and analysis

---

*Code: [github.com/AdamNaghs/models](https://github.com/AdamNaghs/models)*
