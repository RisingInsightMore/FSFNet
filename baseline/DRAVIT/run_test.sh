#!/bin/bash
set -e

# Resolve script directory to absolute path (before any cd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATAROOT="${SCRIPT_DIR}/Datasets/AVIID"
NAME="DRAVIT_AVIID"
GPU_IDS="0"
EPOCH="400"
RESULTS_DIR="${SCRIPT_DIR}/../results"

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

# Ensure DATAROOT is absolute (handles user-supplied relative paths too)
DATAROOT="$(cd "$DATAROOT" 2>/dev/null && pwd)" || { echo "ERROR: DATAROOT '$DATAROOT' does not exist"; exit 1; }

# Convert epoch to 5-digit padded format
EPOCH_PADDED=$(printf "%05d" $EPOCH)

echo "=== DRAVIT Testing ==="
echo "Dataset: $DATAROOT"
echo "Experiment: $NAME"

cd "$SCRIPT_DIR/DR-AVIT/src"
# NOTE: GPU selection is done by passing the PHYSICAL GPU id directly to python
# via `--gpu "$GPU_IDS"`. We intentionally DO NOT `export CUDA_VISIBLE_DEVICES`
# here. DRAVIT's model.py uses `net.cuda(opts.gpu)` with the raw integer as a
# PHYSICAL device id and has NO DataParallel. Setting CUDA_VISIBLE_DEVICES would
# remap devices (e.g. physical 2 becomes visible id 0) and then `.cuda(2)` raises
# `invalid device ordinal`. So we pass the physical id and let all GPUs stay visible.

# Resolve checkpoint path. This MUST run AFTER the `cd` above, because the
# checkpoint lives at ../results (relative to DR-AVIT/src == DR-AVIT/results),
# NOT baseline/results. DRAVIT's saver names checkpoints with the 0-indexed epoch
# number (e.g. 400 training epochs -> 00399.pth), so a --epoch equal to the
# training count is off-by-one and will not exist. If the requested file is
# missing, fall back to the latest numbered checkpoint in the experiment directory.
CKPT="../results/${NAME}/${EPOCH_PADDED}.pth"
if [ ! -f "$CKPT" ]; then
    LATEST=$(ls -1 "../results/${NAME}"/[0-9][0-9][0-9][0-9][0-9].pth 2>/dev/null | sort | tail -n 1)
    if [ -n "$LATEST" ]; then
        CKPT="$LATEST"
        echo "WARN: ${EPOCH_PADDED}.pth not found; using latest checkpoint: $(basename "$LATEST")"
    else
        echo "ERROR: no checkpoint found in ../results/${NAME}/"; exit 1
    fi
fi
CKPT_EPOCH=$(basename "$CKPT" .pth)
echo "Checkpoint: $CKPT"

python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --resume "$CKPT" \
    --gpu "$GPU_IDS"

# Copy results to unified directory
UNIFIED_DIR="${RESULTS_DIR}/${NAME}/test_${CKPT_EPOCH}/images"
mkdir -p "$UNIFIED_DIR"
if [ -d "../outputs/$NAME" ]; then
    cp -r "../outputs/$NAME"/* "$UNIFIED_DIR"/
    echo "Results copied to $UNIFIED_DIR"
fi

echo "=== Testing Complete ==="
