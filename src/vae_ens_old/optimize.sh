#!/bin/bash
#BSUB -J optimize
#BSUB -q gpua100
#BSUB -W 02:00
#BSUB -n 1
#BSUB -R "rusage[mem=6000]"
#BSUB -M 6000
#BSUB -R "span[hosts=1]"
#BSUB -N s204088@dtu.dk
#BSUB -o logs/optimize_%J_%I.out
#BSUB -e logs/optimize_%J_%I.err

module load cuda/11.7
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1
source /dtu/blackhole/1d/155613/venv_geometry/bin/activate

export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface" "logs"

export PYTHONPATH=$PWD:$PYTHONPATH

# Optimize all splines: decoders=1,2,3 with reruns 0–9 and global seed 12
python -m src.optimize_ensemble --global_seed 12 --pairfile selected_pairs_10.json --num_decoders 1 --reruns 0 1 2 3 4 5 6 7 8 9
python -m src.optimize_ensemble --global_seed 12 --pairfile selected_pairs_10.json --num_decoders 2 --reruns 0 1 2 3 4 5 6 7 8 9
python -m src.optimize_ensemble --global_seed 12 --pairfile selected_pairs_10.json --num_decoders 3 --reruns 0 1 2 3 4 5 6 7 8 9
