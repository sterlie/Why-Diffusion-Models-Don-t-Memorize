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

# Clear all inherited modules (incl. gpaw_env) then load Python.
module purge
module load python3/3.10.13

# Change to project directory
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

# Create conda environment if it doesn't exist, then activate it.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f "$script_dir/Experiments/environment_cpu.yml" || true
conda activate memorization

cd "$script_dir/Experiments/src/Training"
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

