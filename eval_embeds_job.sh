#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --gpus=0
#SBATCH --job-name=EVAl_MBDS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:10:00
#SBATCH --output=output/obtain_embeds_for_eval.out

python eval_embeds.py
