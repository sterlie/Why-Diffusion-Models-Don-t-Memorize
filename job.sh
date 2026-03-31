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
module purge
module load dcc-setup/2023-aug
# Enable debugging (prints commands and stops on errors)
set -x
set -e

# Keep cluster module stack; only ensure Python is loaded.
module load python/3.10.13 || module load python/default

# LS_SUBCWD is set by LSF to the directory where bsub was run.
cd "$LS_SUBCWD" || exit 1

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

cd "$LS_SUBCWD/Experiments"
# Upgrade pip and install dependencies (no cache to save space)
python -m pip install --upgrade pip --no-cache-dir
python -m pip install --no-cache-dir -r requirements.txt

cd src/Training
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

