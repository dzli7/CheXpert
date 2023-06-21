#!/bin/bash

#SBATCH --job-name=14model

#SBATCH --output=/home/ding/%j.out
#SBATCH --output=/home/ding/%j.err

#SBATCH -A CS156b

#SBATCH -t 33:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --mem=128G

#SBATCH --gres=gpu:4

#SBATCH --mail-user=ding@caltech.edu

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load cuda/10.2
eval "$(conda shell.bash hook)"
conda activate /home/ding/miniconda3/envs/cs156b

python model14.py
