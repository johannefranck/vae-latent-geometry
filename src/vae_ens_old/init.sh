#!/bin/bash
#BSUB -J init
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

# Run 
# decoders = 1
python -m src.init_ensemble --global_seed 12 --rerun 0 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 1 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 2 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 3 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 4 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 5 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 6 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 7 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 8 --pairfile selected_pairs_10.json --num_decoders 1
python -m src.init_ensemble --global_seed 12 --rerun 9 --pairfile selected_pairs_10.json --num_decoders 1

# decoders = 2
python -m src.init_ensemble --global_seed 12 --rerun 0 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 1 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 2 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 3 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 4 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 5 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 6 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 7 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 8 --pairfile selected_pairs_10.json --num_decoders 2
python -m src.init_ensemble --global_seed 12 --rerun 9 --pairfile selected_pairs_10.json --num_decoders 2

# decoders = 3
python -m src.init_ensemble --global_seed 12 --rerun 0 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 1 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 2 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 3 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 4 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 5 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 6 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 7 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 8 --pairfile selected_pairs_10.json --num_decoders 3
python -m src.init_ensemble --global_seed 12 --rerun 9 --pairfile selected_pairs_10.json --num_decoders 3

