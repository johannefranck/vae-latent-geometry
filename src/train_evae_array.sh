#!/bin/bash
#BSUB -J evae_train[1-3]
#BSUB -q gpua100
#BSUB -W 01:00
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

DECODER_LIST=(1 2 3)
IDX=$((LSB_JOBINDEX - 1))
NUM_DECODERS=${DECODER_LIST[$IDX]}

echo "Training EVAE with $NUM_DECODERS decoders"

CONFIG_OUT=configs/config_run_d${NUM_DECODERS}.yaml
sed "s/num_decoders:.*/num_decoders: ${NUM_DECODERS}/" configs/config.yaml > $CONFIG_OUT

# Run with -m src.train
python -m src.train $CONFIG_OUT
