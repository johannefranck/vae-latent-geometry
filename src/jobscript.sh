#!/bin/bash
#BSUB -J tauoptimize                  # Job name
#BSUB -q gpua100                      # GPU queue
#BSUB -W 00:10                        # Max runtime (hh:mm)
#BSUB -n 1                            # 1 core
# add email
#BSUB -u j
#BSUB -R "rusage[mem=20000]"         # 20GB memory
#BSUB -R "span[hosts=1]"             # All cores on same host
#BSUB -o logs/tauopt_%J.out          # STDOUT
#BSUB -e logs/tauopt_%J.err          # STDERR

# Load modules
module load cuda/11.7

# Go to your project directory
cd /dtu/blackhole/1d/155613/vae-latent-geometry || exit 1

# Activate virtual environment
source venv_geometry/bin/activate

# Set HOME and prepare cache/config folders
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"

# Optional: set PYTHONPATH if your module structure needs it
export PYTHONPATH=$PWD:$PYTHONPATH

# Run the script
python -m src.optimize_tau_energy
