# Hybrid Outcome-Process Reward Modelling for GRPO

Part 2 coursework submission for CMP030L043 Deep Learning and Generative AI
(MSc Artificial Intelligence, University of Roehampton).

Student: Seif Eldin Tourky — A00093494

## Overview

This project implements and evaluates a hybrid reward function for
reinforcement learning fine-tuning of a small language model. It combines:

- **R_outcome** — binary final-answer correctness on GSM8K
- **R_process** — rule-based arithmetic validity of intermediate reasoning steps

combined as `R_hybrid = α · R_outcome + (1 − α) · R_process`.

Training uses Group Relative Policy Optimisation (GRPO) via HuggingFace TRL,
applied to TinyLlama-1.1B-Chat with LoRA adapters, on a subset of GSM8K.

## Experimental design

- 4 reward-mixing coefficients: α ∈ {1.0 (outcome-only baseline), 0.7, 0.5, 0.3}
- 2 random seeds per condition (42, 7) → 8 training runs
- 200 GSM8K training problems, 80 held-out test problems
- Metrics: final-answer accuracy, step-level validity, reasoning chain length
- Statistics: paired bootstrap (10,000 iterations, 95% CIs)

## Key findings

- **Training-time**: outcome-only reward produced near-zero optimisation
  signal across both seeds (sparse-reward collapse under GRPO group
  normalisation). All hybrid conditions produced substantially larger
  training losses, confirming denser learning signal.
- **Test-time**: no statistically detectable differences between conditions
  on any metric (all 95% CIs include zero) — an honest null result at this
  model/data scale.

## Repository contents

- `grpo_hybrid_reward.ipynb` — full pipeline: setup, reward functions,
  GRPO training for all conditions, evaluation, statistics, and plots
- `requirements.txt` — Python dependencies
- `results/` — evaluation outputs (per-problem JSON) for all 8 runs and
  the results figure

## Reproduction

1. Open the notebook in Google Colab with a T4 GPU runtime
2. Run cells top to bottom (installs dependencies, mounts Drive)
3. Training all 8 conditions takes ~90 GPU-minutes total on a T4
4. Evaluation and statistics cells regenerate the results in `results/`

Seeds are fixed (42, 7 for training; 123 for the test subset) for
reproducibility.

## Hardware

All experiments ran on a single NVIDIA T4 (16 GB) via free-tier Google Colab.
