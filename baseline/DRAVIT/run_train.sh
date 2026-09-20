#!/bin/bash
set -e

# Resolve script directory to absolute path (before any cd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATAROOT="${SCRIPT_DIR}/Datasets/AVIID"
NAME="DRAVIT_AVIID"
GPU_IDS="0"
EPOCHS=400
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataroot) DATAROOT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Ensure DATAROOT is absolute (handles user-supplied relative paths too)
DATAROOT="$(cd "$DATAROOT" 2>/dev/null && pwd)" || { echo "ERROR: DATAROOT '$DATAROOT' does not exist"; exit 1; }

N_EP=$EPOCHS
N_EP_DECAY=$((EPOCHS / 2))

echo "=== DRAVIT Training ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"
echo "Epochs: $N_EP (decay from $N_EP_DECAY)"

cd "$SCRIPT_DIR/DR-AVIT/src"
# NOTE: GPU selection is done by passing the PHYSICAL GPU id directly to python
# via `--gpu "$GPU_IDS"`. We intentionally DO NOT `export CUDA_VISIBLE_DEVICES`
# here. DRAVIT's model.py uses `net.cuda(opts.gpu)` with the raw integer as a
# PHYSICAL device id and has NO DataParallel. Setting CUDA_VISIBLE_DEVICES would
# remap devices (e.g. physical 2 becomes visible id 0) and then `.cuda(2)` raises
# `invalid device ordinal`. So we pass the physical id and let all GPUs stay visible.

# Build optional --resume argument (only when provided)
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --n_ep "$N_EP" \
    --n_ep_decay "$N_EP_DECAY" \
    --gpu "$GPU_IDS" \
    $RESUME_ARG

echo "=== Training Complete ==="
