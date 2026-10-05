#!/bin/bash -l
#SBATCH --job-name=tca_own
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=own_experiment_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export PATH="$HOME/.conda/envs/TTA/bin:$PATH"
export WANDB_MODE=disabled

WORK=/home/woody/rlvl/rlvl174v
cd "$HOME/TCA-main"

mkdir -p results_own/ours_0105
mkdir -p results_own/ours_0175

for corruption in contrast snow brightness
do
    for severity in 1 2 3 4 5
    do
        export CIFAR100C_CORRUPTION=$corruption
        export CIFAR100C_SEVERITY=$severity

        echo "============================================"
        echo "TCA Ours-0.105: $corruption severity $severity"
        echo "============================================"

        python runner.py \
            --datasets cifar100c \
            --data-root "$WORK/CIFAR-100-C" \
            --token_pruning Ours-0.105 \
            2>&1 | tee "results_own/ours_0105/${corruption}_s${severity}.txt"

        echo "============================================"
        echo "TCA Ours-0.175: $corruption severity $severity"
        echo "============================================"

        python runner.py \
            --datasets cifar100c \
            --data-root "$WORK/CIFAR-100-C" \
            --token_pruning Ours-0.175 \
            2>&1 | tee "results_own/ours_0175/${corruption}_s${severity}.txt"
    done
done

echo "ALL OWN-EXPERIMENT RUNS FINISHED"
