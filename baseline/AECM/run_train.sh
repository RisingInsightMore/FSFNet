#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="AECM_AVIID"
GPU_IDS="0"
EPOCHS=400
BATCH_SIZE=16
LOAD_SIZE=256
CROP_SIZE=256

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

N_EPOCHS=$((EPOCHS / 2))
N_EPOCHS_DECAY=$((EPOCHS / 2))

echo "=== AECM Training ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"

# NOTE: GPU selection is done by passing the physical GPU IDs directly to python via
# `--gpu_ids "$GPU_IDS"`. We intentionally DO NOT `export CUDA_VISIBLE_DEVICES` here:
# once set, CUDA remaps visible devices to 0,1,... so a physical id like 2 would become
# invalid for torch.cuda.set_device / DataParallel([2,3]) (invalid device ordinal).
python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --dataset_mode unaligned \
    --model aecm \
    --side_length 7 \
    --lambda_spatial 5.0 \
    --lambda_global 5.0 \
    --atten_layers 1,3,5 \
    --local_nums 64 \
    --lr 0.00001 \
    --n_epochs "$N_EPOCHS" \
    --n_epochs_decay "$N_EPOCHS_DECAY" \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gpu_ids "$GPU_IDS" \
    --save_epoch_freq 100

echo "=== Training Complete ==="
