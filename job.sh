#!/bin/bash
#BSUB -q hpc
#BSUB -J memorization
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 25GB
#BSUB -W 24:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

### ===== JOB COMMANDS =====

# Enable debugging (prints commands and stops on errors)
set -x
set -e

# Clear all inherited modules then load Python.
module unload dcc-setup
module load python/3.11.7

# LS_SUBCWD is set by LSF to the directory where bsub was run.
cd "$LS_SUBCWD" || exit 1

# Init conda from its known base path, create env if needed, then activate.
conda env create -f Experiments/environment_cpu.yml || true
conda activate memorization

cd "$LS_SUBCWD/Experiments/src/Training"
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

