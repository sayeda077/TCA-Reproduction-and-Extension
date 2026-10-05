#!/bin/bash -l
#SBATCH --job-name=cifar100c_full
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=cifar100c_full_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export PATH="$HOME/.conda/envs/TTA/bin:$PATH"
export WANDB_MODE=disabled

WORK=/home/woody/rlvl/rlvl174v
cd "$HOME/TCA-main"

mkdir -p results_cifar100c/clip
mkdir -p results_cifar100c/tca

for corruption in contrast snow brightness
do
    for severity in 1 2 3 4 5
    do
        export CIFAR100C_CORRUPTION=$corruption
        export CIFAR100C_SEVERITY=$severity

        echo "========================================"
        echo "CLIP: $corruption severity $severity"
        echo "========================================"

        python runner_clip.py \
          --datasets cifar100c \
          --data-root "$WORK/CIFAR-100-C" \
          --token_pruning EViT-0.0 \
          2>&1 | tee "results_cifar100c/clip/${corruption}_s${severity}.txt"

        echo "========================================"
        echo "TCA: $corruption severity $severity"
        echo "========================================"

        python runner.py \
          --datasets cifar100c \
          --data-root "$WORK/CIFAR-100-C" \
          --token_pruning Ours-0.035 \
          2>&1 | tee "results_cifar100c/tca/${corruption}_s${severity}.txt"

    done
done

echo "ALL CIFAR-100-C RUNS FINISHED"
