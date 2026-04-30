#!/bin/bash
#BSUB -q  gpuv100
#BSUB -J fmem_milk
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4B]"
#BSUB -M 5GB
#BSUB -W 48:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o fmem_milk%J.out
#BSUB -e fmem_milk%J.err

### ===== JOB COMMANDS =====
module purge
module load dcc-setup/2023-aug

set -x
set -e

cd "$LS_SUBCWD" || exit 1
DATA_DIR="$LS_SUBCWD/Experiments/Data"
IMAGE_PTH="$DATA_DIR/MILK10.pth"

source /zhome/61/d/156689/adlcv/Why-Diffusion-Models-Don-t-Memorize/Experiments/mem/bin/activate

pip install pandas --quiet

cd Experiments/src/Evaluation

for N in 256 512 1024 2048; do
    B=$(( N < 512 ? N : 512 ))
    echo "===== Running fmem for n=$N ====="
    python compute_fmem.py \
        -D MILK10 \
        -n $N \
        -i 0 \
        -s 32 \
        -LR 0.0001 \
        -O Adam \
        -W 32 \
        -B $B \
        --num_classes 6 \
        --Ns 1 \
        --gap_threshold 0.333 \
        --device cpu \
        --image_pth "$IMAGE_PTH"
done

echo "All done!"
