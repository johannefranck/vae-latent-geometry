#!/bin/bash
#BSUB -J optimize
#BSUB -q gpua100
#BSUB -W 01:00
#BSUB -n 1
#BSUB -R "rusage[mem=40000]"
#BSUB -M 40000
#BSUB -R "span[hosts=1]"
#BSUB -N s204088@dtu.dk
#BSUB -o init_spline_%J.out
#BSUB -e init_spline_%J.err

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
python -m src.optimize_ensemble --seed 123 --pairfile selected_pairs_50.json --num_decoders 2
python -m src.optimize_ensemble --seed 123 --pairfile selected_pairs_50.json --num_decoders 3
python -m src.optimize_ensemble --seed 12 --pairfile selected_pairs_50.json --num_decoders 1
python -m src.optimize_ensemble --seed 12 --pairfile selected_pairs_50.json --num_decoders 2
python -m src.optimize_ensemble --seed 12 --pairfile selected_pairs_50.json --num_decoders 3
