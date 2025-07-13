#!/bin/bash
#BSUB -J eval_cv
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 1
#BSUB -R "rusage[mem=40000]"
#BSUB -M 40000
#BSUB -R "span[hosts=1]"
#BSUB -N s204088@dtu.dk

module load cuda/11.7
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1
source /dtu/blackhole/1d/155613/venv_geometry/bin/activate

export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=$PWD:$PYTHONPATH


python -m src.eval --mode cov --pair-count 15 --seeds 12 123 1234 12345 456