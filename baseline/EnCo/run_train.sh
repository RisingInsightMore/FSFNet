#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="EnCo_AVIID"
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

echo "=== EnCo Training ==="
# NOTE: GPU selection via physical `--gpu_ids "$GPU_IDS"`; no `export CUDA_VISIBLE_DEVICES`
# (CUDA remap would invalidate physical ids for set_device / DataParallel).
python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model enco \
    --nce_layers 3,7,13,18,24,28 \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$N_EPOCHS" \
    --n_epochs_decay "$N_EPOCHS_DECAY" \
    --num_threads 0 \
    --lambda_IDT 10 \
    --lambda_NCE 2 \
    --netF mlp_sample_with_DAG \
    --lr_G 5e-5 \
    --lr_F 5e-5 \
    --lr_D 2e-4 \
    --warmup_epochs 20 \
    --flip_equivariance True \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --gpu_ids "$GPU_IDS" \
    --save_epoch_freq 100

echo "=== Training Complete ==="
