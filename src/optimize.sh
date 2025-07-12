#!/bin/bash
#BSUB -J optimize[1-4]
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 1
#BSUB -R "rusage[mem=6000]"
#BSUB -M 6000
#BSUB -R "span[hosts=1]"
#BSUB -N s204088@dtu.dk

module load cuda/11.7
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1
source /dtu/blackhole/1d/155613/venv_geometry/bin/activate

export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=$PWD:$PYTHONPATH

# Define combinations
models=("model_seed12.pt" "model_seed12.pt" "model_seed123.pt" "model_seed123.pt")
inits=("entropy" "euclidean" "entropy" "euclidean")

model=${models[$LSB_JOBINDEX - 1]}
init=${inits[$LSB_JOBINDEX - 1]}

echo "Running optimization for $model with $init init"

python -m src.optimize \
  --model-path experiment/$model \
  --init-type $init \
  --pair-count 133 \
  --steps 10 \
  --batch-size 200 \
  --mc-samples 2