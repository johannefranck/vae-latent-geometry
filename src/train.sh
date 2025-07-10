#!/bin/bash
#BSUB -J train_epoch200
#BSUB -q gpua100
#BSUB -W 09:00
#BSUB -n 1
#BSUB -R "rusage[mem=1000]"
#BSUB -M 1000
#BSUB -R "span[hosts=1]"

#BSUB -N s204088@dtu.dk

# Load modules
module load cuda/11.7

# Go to project directory
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1

# Activate virtual environment
source /dtu/blackhole/1d/155613/venv_geometry/bin/activate

# Set HOME and prepare cache/config folders
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"

export PYTHONPATH=$PWD:$PYTHONPATH

# Run single decoder pipeline (batched)
python -m src.train_v101
