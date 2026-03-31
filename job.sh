#!/bin/bash
#BSUB -q hpc
#BSUB -J memorization
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=100GB]"
#BSUB -M 100GB
#BSUB -W 24:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

### ===== JOB COMMANDS =====

# Enable debugging (prints commands and stops on errors)
set -x
set -euo pipefail

# Load Python module available on DTU HPC
module load python/3.11.7

# Change to project directory
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

# Initialize conda for a non-interactive LSF job shell.
if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
else
	for conda_root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
		if [ -f "$conda_root/etc/profile.d/conda.sh" ]; then
			. "$conda_root/etc/profile.d/conda.sh"
			break
		fi
	done
fi

if ! command -v conda >/dev/null 2>&1; then
	echo "conda was not found in the batch environment"
	exit 127
fi

if ! conda run -n memorization python -c "import sys" >/dev/null 2>&1; then
	conda env create -f "$script_dir/Experiments/environment_cpu.yml"
fi

conda activate memorization

cd "$script_dir/Experiments/src/Training"
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1

