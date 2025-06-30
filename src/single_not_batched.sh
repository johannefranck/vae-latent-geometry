#!/bin/bash
#BSUB -J geodesic_single_decoder      # Job name
#BSUB -q gpua100                      # GPU queue
#BSUB -W 00:12                        # Max runtime (hh:mm)
#BSUB -n 1                            # 1 core
#BSUB -B s204088@dtu.dk               # Send email at start
#BSUB -N s204088@dtu.dk               # Send email at end of job
#BSUB -R "rusage[mem=20000]"         # 20GB memory
#BSUB -R "span[hosts=1]"             # All cores on same host
#BSUB -o single.out                  # STDOUT
#BSUB -e single.err                  # STDERR

# Load modules
module load cuda/11.7

# Go to your project directory
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1

# Activate virtual environment
source /dtu/blackhole/1d/155613/venv_geometry/bin/activate

# Set HOME and prepare cache/config folders
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"

export PYTHONPATH=$PWD:$PYTHONPATH

# Run single decoder pipeline (none-batched)
python -m src.single_decoder.init_spline --seed 12
python -m src.single_decoder.optimize_energy --seed 12
python -m src.single_decoder.density --seed 12
