#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="Pix2Pix_AVIID"
GPU_IDS="0"
EPOCHS=400
BATCH_SIZE=32
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

echo "=== Pix2Pix Training ==="

# Pix2Pix aligned mode requires concatenated AB images in train/ and test/.
# Preprocess: merge trainA+trainB -> train/, testA+testB -> test/ (skip if done).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/prep_pix2pix.py" --dataroot "$DATAROOT"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model pix2pix \
    --direction AtoB \
    --dataset_mode aligned \
    --n_epochs "$EPOCHS" \
    --n_epochs_decay 0 \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --save_epoch_freq 100

echo "=== Training Complete ==="
