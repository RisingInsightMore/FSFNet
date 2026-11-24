#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="CycleGAN_AVIID"
GPU_IDS="0"
EPOCHS=400
BATCH_SIZE=8
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

echo "=== CycleGAN Training ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"
echo "GPU: $GPU_IDS"
echo "Epochs: $EPOCHS"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model cycle_gan \
    --pool_size 50 \
    --no_dropout \
    --n_epochs "$EPOCHS" \
    --n_epochs_decay 0 \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --save_epoch_freq 100

echo "=== Training Complete ==="
