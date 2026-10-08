# TCA Reproduction and Experimental Extension

This repository contains a reproduction and experimental extension of **Token Condensation as Adaptation (TCA)** for efficient test-time adaptation of vision-language models.

## Project Overview

This project has two main parts:

1. **Reproduction:** Reproduce selected TCA experiments and compare TCA with baseline methods.
2. **Experimental Extension:** Investigate how different TCA pruning strengths affect classification accuracy and computational cost.

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
│   │   ├── ours_0105/
│   │   └── ours_0175/
│   ├── profile_flops_own.py
│   ├── results_flops_own.txt
│   └── run_own_experiment.sh
│
├── .gitignore
└── README.md
```

## Part 1: Reproduction

The first part reproduces selected experiments from the TCA work.

The reproduction includes:

- Cross-dataset evaluation
- CIFAR-100-C robustness evaluation
- Comparison of CLIP, EViT, ToMe, and TCA
- Computational cost comparison using GFLOPs

For the cross-dataset evaluation, the reproduced TCA result achieved an average accuracy of approximately **68.30%**.

For the CIFAR-100-C robustness evaluation, three corruption types were selected:

- Contrast
- Snow
- Brightness

Each corruption was evaluated at severity levels **1–5**.

For these 15 selected corruption/severity combinations, the reproduced results were:

| Method | Mean Accuracy |
|---|---:|
| CLIP | 58.10% |
| TCA (Ours-0.035) | 58.64% |

These results were used as the baseline for the experimental extension.

## Part 2: Experimental Extension

The second part investigates what happens when stronger TCA token-reduction settings are used.

Three settings were compared:

| TCA Setting | Mean Accuracy | GFLOPs | Compute Saving |
|---|---:|---:|---:|
| Ours-0.035 | 58.64% | 15.710 | 10.65% |
| Ours-0.105 | 52.33% | 11.683 | 33.55% |
| Ours-0.175 | 34.71% | 8.826 | 49.80% |

The results show a clear **accuracy-efficiency trade-off**.

Increasing the token-reduction strength reduces computational cost, but stronger reduction also causes a substantial decrease in classification accuracy.

The **Ours-0.105** setting provides a useful intermediate point: it reduces computation by approximately **33.55%** while maintaining considerably higher accuracy than the more aggressive Ours-0.175 setting.

## Main Finding

The experiment demonstrates that maximizing computational savings is not necessarily the best choice.

More aggressive token reduction:

- reduces GFLOPs,
- increases computational savings,
- but can significantly reduce classification accuracy.

Therefore, the token-condensation strength should be selected by balancing **accuracy and computational efficiency** rather than focusing only on maximum compute reduction.

## Experimental Setup

- **Model:** CLIP ViT-B/16
- **Robustness Dataset:** CIFAR-100-C
- **Corruptions:** Contrast, Snow, Brightness
- **Severity Levels:** 1–5
- **TCA Settings:** Ours-0.035, Ours-0.105, Ours-0.175
- **Compute Metric:** GFLOPs
- **Computing Environment:** NHR@FAU HPC
- **Job Scheduler:** Slurm

## Results

The raw reproduction outputs are stored in:

```text
reproduction/results/
```

The additional experimental results are stored in:

```text
own_experiment/results/
```

The extension contains results for:

```text
ours_0105/
ours_0175/
```

Each setting was evaluated on:

```text
Contrast   → Severity 1–5
Snow       → Severity 1–5
Brightness → Severity 1–5
```

The GFLOPs measurements for the extension are stored in:

```text
own_experiment/results_flops_own.txt
```

## Running the Experiments

The reproduction scripts are available in the `reproduction/` directory.

The experimental extension can be launched using:

```bash
bash own_experiment/run_own_experiment.sh
```

Dataset paths, Slurm configuration, and computing resources may need to be changed when running the experiments on another system.

## Reproducibility

This repository provides the main materials used for the project, including:

- Source code
- Experiment scripts
- Configuration files
- Raw experimental outputs
- CIFAR-100-C result files
- GFLOPs measurements

Results may vary slightly depending on hardware, software versions, dataset configuration, and runtime environment.

## Original TCA Code

The directory:

```text
original_tca_code/
```

contains the original TCA implementation used as the basis for the reproduction.

The original TCA implementation is kept separate from:

```text
reproduction/
own_experiment/
```

to clearly distinguish the original implementation from the reproduction work and the additional experiments performed in this project.

**The original TCA code is not claimed as original work by the author of this repository. Credit belongs to the authors of the original TCA work.**

