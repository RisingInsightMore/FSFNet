#!/bin/bash
# 非主入口辅助脚本：与 run_train.sh 保持一致，默认物理 GPU 0，
# 可用环境变量 GPU_IDS 覆盖（如 GPU_IDS="2" ./train_backward.sh）。
GPU_IDS="${GPU_IDS:-0}"
python train.py --name your_experiment_name --dataroot /your/dataset/path --model cyclemamba1 --direction AtoB --pretrain_path None --gpu_ids "$GPU_IDS"
