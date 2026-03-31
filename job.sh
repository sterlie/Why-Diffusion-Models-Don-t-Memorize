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

# Load Python/conda module on the cluster.
module load python/3.11.7

# Change to project directory
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

# Activate an existing conda environment in a non-interactive shell.
eval "$(conda shell.bash hook)"
conda activate memorization

cd "$script_dir/Experiments/src/Training"
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

