#!/bin/bash
#BSUB -J optimize50
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 1
#BSUB -R "rusage[mem=6000]"
#BSUB -M 6000
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

# Run optimize
python -m src.optimize
