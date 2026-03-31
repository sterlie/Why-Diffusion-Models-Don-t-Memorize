#!/bin/bash
#BSUB -q hpc
#BSUB -J memorization
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=100GB]"
#BSUB -M 100GB
#BSUB -W 24:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Error_%J.err

### ===== JOB COMMANDS =====

# Enable debugging (prints commands and stops on errors)
set -x
set -e

# Load Python module available on DTU HPC
module load python/3.11.7

# Change to project directory
cd /zhome/61/d/156689/adlcv|| exit 1

# Create logs directory if it does not exist
mkdir -p logs

conda env create -f environment_cpu.yml
conda activate memorization
pip install natsort

cd Experiments/src/Training
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

