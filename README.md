# TCA Reproduction and Extension

This repository contains a reproduction and experimental extension of **Token Condensation as Adaptation (TCA)** for efficient test-time adaptation of vision-language models.

The project was completed as part of the **AI Systems Project at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**.

## Project Overview

The project has two main parts:

1. **Reproduction:** Reproduce selected experiments from the TCA work and compare TCA with baseline methods.
2. **Extension:** Study how different TCA pruning strengths affect the trade-off between classification accuracy and computational cost.

The experiments use **CLIP ViT-B/16** and include cross-dataset evaluation and robustness experiments on **CIFAR-100-C**.

## Repository Structure

```text
TCA-Reproduction-and-Extension/
│
├── original_tca_code/
│   └── Original TCA implementation used as the basis of this project
│
├── reproduction/
│   ├── results/
│   ├── profile_flops.py
│   ├── run_cifar100c_full.sh
│   └── run_tca_remaining.sh
│
├── own_experiment/
│   ├── results/
│   ├── profile_flops_own.py
│   ├── results_flops_own.txt
│   └── run_own_experiment.sh
│
└── README.md

