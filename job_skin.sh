#!/bin/bash
#BSUB -q  gpuv100
#BSUB -J skin_all_samples
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=9GB]"
#BSUB -M 10GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o skin_all_%J.out
#BSUB -e skin_all_%J.err

### ===== JOB COMMANDS =====
module purge
module load dcc-setup/2023-aug
# Enable debugging (prints commands and stops on errors)
set -x
set -e

# Set dataset size to train — change before submitting
N= 10480

cd "$LS_SUBCWD" || exit 1
DATA_DIR="$LS_SUBCWD/Experiments/Data"

# Activate project venv (has compatible PyTorch for V100 CC 7.0)
source /zhome/61/d/156689/adlcv/Why-Diffusion-Models-Don-t-Memorize/Experiments/mem/bin/activate

# Install missing dependencies
pip install pandas --quiet

cd Experiments/src/Training

# Optional (only matters if GPU is used)
python -c "import torch; print(torch.cuda.is_available())"

# Run with safer memory settings
python run_Unet_guided.py \
  -n $N \
  -s 32 \
  -W 32 \
  -LR 0.0001 \
  -O Adam \
  -m "$DATA_DIR/MILK10k_Training_Metadata.csv" \
  -p "$DATA_DIR/MILK10.pth" \
  -l skin_tone_class \
  --device cuda:0 \
  --generate