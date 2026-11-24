#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="CycleMamba_AVIID"
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

echo "=== CycleMamba Training (Two-Stage) ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"

# 重要：不再 export CUDA_VISIBLE_DEVICES，而是把“物理卡号”直接传给 --gpu_ids。
# 原因：模型由 networks.init_net 用 net.to(gpu_ids[0]) + torch.nn.DataParallel(net, gpu_ids)
# 放置于 GPU，gpu_ids 必须是“真实可见设备序号”。若同时 export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# 并传同样的 --gpu_ids，逻辑序号会被重映射（2,3 -> 0,1），导致 DataParallel 报
# invalid device ordinal。故移除 export，直接传物理卡号端到端生效、多卡可用。

# Stage 1: Backward training
echo ""
echo "=== Stage 1: Backward Training ==="
python train.py \
    --name "${NAME}_backward" \
    --dataroot "$DATAROOT" \
    --model cyclemamba1 \
    --direction AtoB \
    --pretrain_path None \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$EPOCHS" \
    --n_epochs_decay 0 \
    --gpu_ids "$GPU_IDS" \
    --dataset_mode unaligned

# Get Stage 1 checkpoint
STAGE1_CKPT="./checkpoints/${NAME}_backward/latest_net_G.pth"
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    exit 1
fi

# Stage 2: Forward training
echo ""
echo "=== Stage 2: Forward Training ==="
python my_train.py \
    --name "$NAME" \
    --dataroot "$DATAROOT" \
    --model cyclemamba2 \
    --direction AtoB \
    --pretrain_path "$STAGE1_CKPT" \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$EPOCHS" \
    --n_epochs_decay 0 \
    --gpu_ids "$GPU_IDS" \
    --dataset_mode unaligned

echo "=== Training Complete ==="
