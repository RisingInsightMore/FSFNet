#!/bin/bash
set -e

DATAROOT="./datasets/AVIID"
NAME="Pix2Pix_AVIID"
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

echo "=== Pix2Pix Testing ==="

# Pix2Pix aligned mode requires concatenated AB images in test/.
python -c "
import os, glob, numpy as np
from PIL import Image

dataroot = '$DATAROOT'
for phase in ['train', 'test']:
    dir_out = os.path.join(dataroot, phase)
    dir_A = os.path.join(dataroot, phase + 'A')
    dir_B = os.path.join(dataroot, phase + 'B')
    if os.path.isdir(dir_out) and len(os.listdir(dir_out)) > 0:
        continue
    if not os.path.isdir(dir_A) or not os.path.isdir(dir_B):
        continue
    os.makedirs(dir_out, exist_ok=True)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    paths_A = sorted([p for e in exts for p in glob.glob(os.path.join(dir_A, e))])
    count = 0
    for pA in paths_A:
        fname = os.path.basename(pA)
        pB = os.path.join(dir_B, fname)
        if not os.path.isfile(pB):
            continue
        im_A = np.array(Image.open(pA).convert('RGB'))
        im_B = np.array(Image.open(pB).convert('RGB'))
        if im_A.shape != im_B.shape:
            im_B = np.array(Image.fromarray(im_B).resize((im_A.shape[1], im_A.shape[0]), Image.BICUBIC))
        concat = np.concatenate([im_A, im_B], axis=1)
        Image.fromarray(concat).save(os.path.join(dir_out, fname))
        count += 1
    if count > 0:
        print(f'[prep] Created {count} paired images in {dir_out}')
"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model pix2pix \
    --direction AtoB \
    --dataset_mode aligned \
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
