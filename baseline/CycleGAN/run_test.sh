#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="CycleGAN_AVIID"
GPU_IDS="0"
EPOCH="400"
NUM_TEST="10000"
RESULTS_DIR="./results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --num_test) NUM_TEST="$2"; shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== CycleGAN Testing ==="

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model cycle_gan \
    --load_size 256 \
    --crop_size 256 \
    --epoch "$EPOCH" \
    --num_test "$NUM_TEST" \
    --results_dir "$RESULTS_DIR"

# Copy results to unified directory
UNIFIED_DIR="../results/${NAME}/test_${EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
SRC_DIR=$(find ./results/$NAME -type d -name "images" | head -1)
if [ -n "$SRC_DIR" ]; then
    cp -r "$SRC_DIR"/* "$UNIFIED_DIR"/
    echo "Results copied to $UNIFIED_DIR"
fi

echo "=== Testing Complete ==="
