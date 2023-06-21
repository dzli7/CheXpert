#!/bin/bash

#SBATCH --job-name=test

#SBATCH --output=/home/ding/%j.out
#SBATCH --output=/home/ding/%j.err

#SBATCH -A CS156b

#SBATCH -t 00:01:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=1M

#SBATCH --mail-user=ding@caltech.edu

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python3/3.8.5

source /home/ding/venvs/cs156b/bin/activate

python test.py
