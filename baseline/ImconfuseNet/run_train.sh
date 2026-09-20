#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="ImconfuseNet_AVIID"
GPU_IDS="0"
EPOCHS=400
BATCH_SIZE=4
CROP_SIZE=128
IMAGE_SIZE=128
SR_H=128
SR_W=128
CONTINUE_TRAIN=""
EPOCH="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --continue_train) CONTINUE_TRAIN="--continue_train"; shift ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --dataset)
            case "$2" in
                AVIID) DATAROOT="./datasets/AVIID"; NAME="ImconfuseNet_AVIID" ;;
                DayDrone) DATAROOT="./datasets/DayDrone"; NAME="ImconfuseNet_DayDrone" ;;
                NightDrone) DATAROOT="./datasets/NightDrone"; NAME="ImconfuseNet_NightDrone" ;;
                *) echo "Unknown dataset: $2"; exit 1 ;;
            esac
            shift 2 ;;
        *) shift ;;
    esac
done

echo "=== ImconfuseNet Training ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"
echo "GPU: $GPU_IDS"
echo "Epochs: $EPOCHS"
echo "Resume from checkpoint: $EPOCH"
if [ -n "$CONTINUE_TRAIN" ]; then
    echo "Continue training: yes"
else
    echo "Continue training: no"
fi

if [ ! -d "$DATAROOT" ]; then
    echo "ERROR: Dataset directory not found: $DATAROOT"
    echo "Please create dataset symlinks first."
    exit 1
fi

# NOTE: GPU selection is done by passing Physical GPU IDs directly to python via
# `--gpu_ids "$GPU_IDS"`. We intentionally DO NOT `export CUDA_VISIBLE_DEVICES`
# here. Once CUDA_VISIBLE_DEVICES is set, CUDA remaps the visible devices to
# 0,1,..., so a physical id like 2 would become invalid and torch.cuda.set_device
# / DataParallel would raise "invalid device ordinal" or pick the wrong card.
# python (options/base_options.py) uses the physical ids as-is for
# torch.cuda.set_device(opt.gpu_ids[0]) and DataParallel(net, gpu_ids).

python -u train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model cycle_gan_hsi_INR_cycle \
    --sr_H "$SR_H" \
    --sr_W "$SR_W" \
    --image_size "$IMAGE_SIZE" \
    --serial_batches \
    --no_flip \
    --preprocess resize \
    --batch_size "$BATCH_SIZE" \
    --crop_size "$CROP_SIZE" \
    --n_epochs "$EPOCHS" \
    --n_epochs_decay 0 \
    --gpu_ids "$GPU_IDS" \
    $CONTINUE_TRAIN \
    --epoch "$EPOCH"

echo "=== Training Complete ==="
