#!/bin/bash
#BSUB -q hpc
#BSUB -J memorization
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 25GB
#BSUB -W 100:00
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
python -c 
"import torch; print(torch.cuda.is_available())"

# Run with safer memory settings

python run_Unet_guided.py  --num 128 --batch_size 16 --label_col skin_tone_class --metadata_csv ../../Data/milk10/MILK10k_Training_Metadata.csv --image_pth ../../Data/milk10/MILK10k_Training_Input.pth  --img_size 32 --learning_rate 0.0001 --optim Adam --nbase 12