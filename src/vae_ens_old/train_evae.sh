#!/bin/bash
#BSUB -J evae_train
#BSUB -q gpua100
#BSUB -W 02:00
#BSUB -n 1
#BSUB -R "rusage[mem=6000]"
#BSUB -M 6000
#BSUB -R "span[hosts=1]"
#BSUB -N s204088@dtu.dk
#BSUB -o logs/train_%J_%I.out
#BSUB -e logs/train_%J_%I.err

module load cuda/11.7
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1
source /dtu/blackhole/1d/155613/venv_geometry/bin/activate

export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface" "logs"

export PYTHONPATH=$PWD:$PYTHONPATH

# Run with -m src.train
python -m src.train_evae configs/config_train_evae.yaml
