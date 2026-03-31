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

# Keep cluster module stack; only ensure Python is loaded.
module load python/3.11.7 || module load python/default

# LS_SUBCWD is set by LSF to the directory where bsub was run.
cd "$LS_SUBCWD" || exit 1

# Initialize conda for this non-interactive shell.
if [ -f /zhome/projects/k10240/gpaw_env/etc/profile.d/conda.sh ]; then
	source /zhome/projects/k10240/gpaw_env/etc/profile.d/conda.sh
fi
conda env create -f Experiments/environment_cpu.yml || true
conda activate memorization

cd "$LS_SUBCWD/Experiments/src/Training"
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

