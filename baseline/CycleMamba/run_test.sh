#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="CycleMamba_AVIID"
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

echo "=== CycleMamba Testing ==="
# 不再 export CUDA_VISIBLE_DEVICES：直接把“物理卡号”传给 --gpu_ids，
# 由 networks.init_net 用 DataParallel(gpu_ids) 放置于对应物理 GPU。
python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model cyclemamba2 \
    --direction AtoB \
    --pretrain_path "./checkpoints/${NAME}_backward/latest_net_G.pth" \
    --load_size 256 \
    --crop_size 256 \
    --gpu_ids "$GPU_IDS" \
    --dataset_mode unaligned \
    --num_test "$NUM_TEST"

UNIFIED_DIR="../results/${NAME}/test_${EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
SRC_DIR=$(find ./results/$NAME -type d -name "images" | head -1)
if [ -n "$SRC_DIR" ]; then
    cp -r "$SRC_DIR"/* "$UNIFIED_DIR"/
fi

echo "=== Testing Complete ==="
