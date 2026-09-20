#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="ImconfuseNet_AVIID"
GPU_IDS="0"
EPOCH="latest"
NUM_TEST="10000"
RESULTS_DIR="./results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
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

echo "=== ImconfuseNet Testing ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"
echo "Epoch: $EPOCH"

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

python -u test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model cycle_gan_hsi_INR_cycle \
    --sr_H 128 \
    --sr_W 128 \
    --image_size 128 \
    --epoch "$EPOCH" \
    --gpu_ids "$GPU_IDS" \
    --num_test "$NUM_TEST" \
    --serial_batches \
    --no_flip \
    --preprocess resize

UNIFIED_DIR="../results/${NAME}/test_${EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
SRC_DIR=$(find ./results/$NAME -type d -name "images" | head -1)
if [ -n "$SRC_DIR" ]; then
    cp -r "$SRC_DIR"/* "$UNIFIED_DIR"/
    echo "Results copied to $UNIFIED_DIR"
else
    echo "WARNING: No results directory found"
fi

echo "=== Testing Complete ==="
