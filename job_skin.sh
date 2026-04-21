#!/bin/bash
#BSUB -q hpc
#BSUB -J guided_skin 
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 25GB
#BSUB -W 48:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o skin_%J.out
#BSUB -e skin_%J.err

### ===== JOB COMMANDS =====
module purge
module load dcc-setup/2023-aug
# Enable debugging (prints commands and stops on errors)
set -x
set -e

cd "$LS_SUBCWD" || exit 1
DATA_DIR="$LS_SUBCWD/Experiments/Data"

# Activate prebuilt environment
source ~/gpaw_env/bin/activate

# Install missing dependencies
pip install pandas --quiet

cd Experiments/src/Training

# Optional (only matters if GPU is used)
python -c "import torch; print(torch.cuda.is_available())"

# Run with safer memory settings
python run_Unet_guided.py \
  -n 1024 \
  -b 64 \
  -s 32 \
  -W 128 \
  -LR 0.0001 \
  -O Adam \
  -m "$DATA_DIR/MILK10k_Training_Metadata.csv" \
  -p "$DATA_DIR/MILK10k_Training_Input.pth" \
  -l skin_tone_class \
  --device cpu \
  --generate