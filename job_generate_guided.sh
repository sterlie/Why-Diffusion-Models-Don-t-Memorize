#!/bin/bash
#BSUB -q gpuv100
#BSUB -J gen_guided
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -M 7GB
#BSUB -W 48:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o gen_guided_%J.out
#BSUB -e gen_guided_%J.err

### ===== JOB COMMANDS =====
# Submits one job per class (LSF job array index = class label).
# Submit with: bsub job_generate_guided.sh
# Each task generates samples for class $LSB_JOBINDEX (0-5).

module purge
module load dcc-setup/2023-aug

set -x
set -e

cd "$LS_SUBCWD" || exit 1
DATA_DIR="$LS_SUBCWD/Experiments/Data"

source /zhome/61/d/156689/adlcv/Why-Diffusion-Models-Don-t-Memorize/Experiments/mem/bin/activate

pip install pandas --quiet

cd Experiments/src/Generation

python -c "import torch; print(torch.cuda.is_available())"

for N in 256 512 1024 2048; do
  B=$(( N < 512 ? N : 512 ))
  echo "===== Generating for n=$N class=0 ====="
  python generate.py \
    -D MILK10 \
    -n $N \
    -i 0 \
    -s 32 \
    -B $B \
    -LR 0.0001 \
    -O Adam \
    -W 32 \
    -Ns 100 \
    --device cuda:0 \
    --num_classes 6 \
    --class_label 0 \
    --available_only
done