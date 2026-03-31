#!/bin/bash
#BSUB -q hpc
#BSUB -J memorization
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 25GB
#BSUB -W 24:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

### ===== JOB COMMANDS =====
module purge
module load dcc-setup/2023-aug
# Enable debugging (prints commands and stops on errors)
set -x
set -e

cd "$LS_SUBCWD" || exit 1

# Activate prebuilt environment
source ~/gpaw_env/bin/activate

cd Experiments/src/Training

# Optional (only matters if GPU is used)
python -c "import torch; print(torch.cuda.is_available())"

# Run with safer memory settings
python run_GMM.py -n 2048 -d 8 -s 1 -de 128 -O Adam -B 128 -t -1