#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="CUT_AVIID"
GPU_IDS="0"
EPOCHS=400
EPOCH_COUNT=1
CHECKPOINT_EPOCH="latest"   # 续训时加载的检查点：latest | 100 | 200 | ...
BATCH_SIZE=8
LOAD_SIZE=256
CROP_SIZE=256
RESUME=""   # 非空（传 --resume）时开启断点续训，从 CHECKPOINT_EPOCH 加载

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --epoch_count) EPOCH_COUNT="$2"; shift 2 ;;
        --checkpoint_epoch) CHECKPOINT_EPOCH="$2"; shift 2 ;;
        --resume) RESUME="1"; shift ;;
        *) shift ;;
    esac
done

echo "=== CUT Training ==="
echo "dataroot=$DATAROOT name=$NAME epochs=$EPOCHS epoch_count=$EPOCH_COUNT checkpoint_epoch=$CHECKPOINT_EPOCH resume=${RESUME:-no}"

# 重要：不再设置 CUDA_VISIBLE_DEVICES，而是把“物理卡号”直接传给 --gpu_ids。
# 原因：原脚本只 export CUDA_VISIBLE_DEVICES="$GPU_IDS" 却不给 train.py 传 --gpu_ids，
# 导致 train.py 用默认 gpu_ids=0，在 CUDA_VISIBLE_DEVICES 未生效时被放到错误的物理 GPU 上。

TRAIN_ARGS=(
    --dataroot "$DATAROOT"
    --name "$NAME"
    --gpu_ids "$GPU_IDS"
    --CUT_mode CUT
    --n_epochs "$EPOCHS"
    --n_epochs_decay 0
    --epoch_count "$EPOCH_COUNT"
    --load_size "$LOAD_SIZE"
    --crop_size "$CROP_SIZE"
    --batch_size "$BATCH_SIZE"
    --save_epoch_freq 100
)
if [[ -n "$RESUME" ]]; then
    TRAIN_ARGS+=(--continue_train --epoch "$CHECKPOINT_EPOCH")
fi

python train.py "${TRAIN_ARGS[@]}"

echo "=== Training Complete ==="
