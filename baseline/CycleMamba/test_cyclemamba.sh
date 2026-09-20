#!/bin/bash
# 非主入口辅助脚本：与 run_test.sh 保持一致，默认物理 GPU 0，
# 可用环境变量 GPU_IDS 覆盖（如 GPU_IDS="2" ./test_cyclemamba.sh）。
GPU_IDS="${GPU_IDS:-0}"
python test.py --dataroot /dataset/path --name /your/test/name --model cyclemamba2 --gpu_ids "$GPU_IDS" --direction AtoB --pretrain_path /pretrain_path #--epoch latest --pre_epoch latest
