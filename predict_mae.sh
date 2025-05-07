#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=MAE_ECG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:10:00
#SBATCH --output=output/pre_trained_ECG_results.out

source activate mae3

python predict_ECG.py