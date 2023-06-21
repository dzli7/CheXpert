#!/bin/bash

#SBATCH --job-name=data

#SBATCH --output=/home/ding/%j.out
#SBATCH --output=/home/ding/%j.err

#SBATCH -A CS156b

#SBATCH -t 00:02:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=3G

#SBATCH --gres=gpu:1

#SBATCH --mail-user=ding@caltech.edu

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load cuda/10.2
eval "$(conda shell.bash hook)"
conda activate /home/ding/miniconda3/envs/cs156b

python access_data.py
