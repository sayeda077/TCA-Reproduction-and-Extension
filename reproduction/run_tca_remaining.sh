#!/bin/bash -l
#SBATCH --job-name=tca_r09_rem
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=tca_r09_rem_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export PATH="$HOME/.conda/envs/TTA/bin:$PATH"
cd "$HOME/TCA-main"

( time python runner.py \
  --datasets fgvc/food101/oxford_pets/stanford_cars/sun397/ucf101 \
  --data-root data/ \
  --token_pruning Ours-0.035 ) \
  2>&1 | tee results_TCA_R0.9_remaining.txt
