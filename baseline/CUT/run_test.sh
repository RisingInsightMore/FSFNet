#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="CUT_AVIID"
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

echo "=== CUT Testing ==="

# 重要：不再设置 CUDA_VISIBLE_DEVICES，而是把“物理卡号”直接传给 --gpu_ids。
# 原因：原脚本只 export CUDA_VISIBLE_DEVICES="$GPU_IDS" 却不给 test.py 传 --gpu_ids，
# 导致 test.py 用默认 gpu_ids=0，在 CUDA_VISIBLE_DEVICES 未生效时被放到错误的物理 GPU 上。
python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --gpu_ids "$GPU_IDS" \
    --CUT_mode CUT \
    --load_size 256 \
    --crop_size 256 \
    --epoch "$EPOCH" \
    --num_test "$NUM_TEST" \
    --results_dir "$RESULTS_DIR"

UNIFIED_DIR="../results/${NAME}/test_${EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
SRC_DIR=$(find ./results/$NAME -type d -name "images" | head -1)
if [ -n "$SRC_DIR" ]; then
    cp -r "$SRC_DIR"/* "$UNIFIED_DIR"/
fi

echo "=== Testing Complete ==="
