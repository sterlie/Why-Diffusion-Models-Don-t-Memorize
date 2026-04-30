#BSUB -q gpuv100
#BSUB -J gen_guided_6
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 4:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o gen_guided_6_.out
#BSUB -e gen_guided_6_.err

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

for N in 512, 1024, 2048; do
  echo "===== Running fmem for n=$N ====="
  python generate.py \
    -D MILK10 \
    -n $N \
    -i 0 \
    -s 32 \
    -B 512 \
    -LR 0.0001 \
    -O Adam \
    -W 128 \
    -Ns 100 \
    --device cpu \
    --num_classes 6 \
    --class_label 0 \
    --available_only
done