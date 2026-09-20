# evaluate_ustnet.py
# 统一红外(可见光->红外)翻译指标评估入口（与 AECM 口径严格一致）。
# 用法:
#   cd ./baseline/USTNet
#   python evaluate_ustnet.py --input ./results/USTNet_AVIID/test_400/images \
#                             --out ./eval_USTNet_AVIID --fid
# 说明: 默认算 LPIPS/SSIM/PSNR/RMSE；加 --fid 额外算 FID/KID。
#       依赖 results 目录下同时存在 fake_B/ 与 real_B/ 子目录（USTNet 的 visualizer 默认产出）。
#       注: USTNet 原有 evaluation_metric.py（接口为 --real/--fake 分开且依赖项目 config、无 FID），
#           本脚本提供更统一的 --input/--out/--fid 入口，二者可并存。
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from eval_core import run_evaluation

if __name__ == '__main__':
    run_evaluation('USTNet')
