#!/bin/bash
#BSUB -J init
#BSUB -q gpua100
#BSUB -W 01:00
#BSUB -n 1
#BSUB -R "rusage[mem=6000]"
#BSUB -M 6000
#BSUB -R "span[hosts=1]"
#BSUB -N s204088@dtu.dk
#BSUB -o logs/optimize_%J_%I.out
#BSUB -e logs/optimize_%J_%I.err

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

# run init, optimize, cov
python -m src.init_ensemble \
    --encoder_seed 12 \
    --rerun 0 \
    --pairfile selected_pairs_10.json \
    --num_decoders 3
python -m src.init_ensemble \
    --encoder_seed 12 \
    --rerun 1 \
    --pairfile selected_pairs_10.json \
    --num_decoders 3
python -m src.init_ensemble \
    --encoder_seed 12 \
    --rerun 2 \
    --pairfile selected_pairs_10.json \
    --num_decoders 3
python -m src.init_ensemble \
    --encoder_seed 12 \
    --rerun 3 \
    --pairfile selected_pairs_10.json \
    --num_decoders 3
python -m src.init_ensemble \
    --encoder_seed 12 \
    --rerun 4 \
    --pairfile selected_pairs_10.json \
    --num_decoders 3

python -m src.optimize_ensemble \
  --encoder_seed 12 \
  --pairfile selected_pairs_10.json \
  --num_decoders 3 \
  --reruns 0 1 2 3 4

python -m src.evaluate_cov \
  --pairfile selected_pairs_10.json \
  --encoder_seed 12 \
  --reruns 0 1 2 3 4 \
  --max_decoders 3