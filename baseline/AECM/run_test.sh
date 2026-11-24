#!/bin/bash
set -e

DATAROOT="./dataset/AVIID"
NAME="AECM_AVIID"
GPU_IDS="0"
EPOCH="latest"
RESULTS_DIR="./results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=== AECM Testing ==="

# NOTE: GPU selection via physical `--gpu_ids "$GPU_IDS"`; no `export CUDA_VISIBLE_DEVICES`
# (CUDA remap would invalidate physical ids for set_device / DataParallel).
python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model aecm \
    --num_test 10000 \
    --epoch "$EPOCH" \
    --gpu_ids "$GPU_IDS"

UNIFIED_DIR="../results/${NAME}/test_${EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
# AECM saves to results/{name}_{model}/test_{epoch}/images/
SRC_DIR=$(find ./results/${NAME}_aecm -type d -name "images" 2>/dev/null | head -1)
if [ -n "$SRC_DIR" ]; then
    cp -r "$SRC_DIR"/* "$UNIFIED_DIR"/
    echo "Results copied to $UNIFIED_DIR"
fi

echo "=== Testing Complete ==="
