#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="EnCo_AVIID"
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

echo "=== EnCo Testing ==="
# NOTE: GPU selection via physical `--gpu_ids "$GPU_IDS"`; no `export CUDA_VISIBLE_DEVICES`
# (CUDA remap would invalidate physical ids for set_device / DataParallel).
python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model enco \
    --phase test \
    --netF mlp_sample_with_DAG \
    --load_size 256 \
    --crop_size 256 \
    --epoch "$EPOCH" \
    --results_dir "$RESULTS_DIR" \
    --gpu_ids "$GPU_IDS" \
    --num_test "$NUM_TEST"

UNIFIED_DIR="../results/${NAME}/test_${EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
SRC_DIR=$(find ./results/$NAME -type d -name "images" | head -1)
if [ -n "$SRC_DIR" ]; then
    cp -r "$SRC_DIR"/* "$UNIFIED_DIR"/
fi

echo "=== Testing Complete ==="
