# evaluate_pix2pix.py
# 统一红外(可见光->红外)翻译指标评估入口（与 AECM 口径严格一致）。
# 用法:
#   cd ./baseline/Pix2Pix
#   python evaluate_pix2pix.py --input ./results/Pix2Pix_AVIID/test_400/images \
#                              --out ./eval_Pix2Pix_AVIID --fid
# 说明: 默认算 LPIPS/SSIM/PSNR/RMSE；加 --fid 额外算 FID/KID。
#       依赖 results 目录下同时存在 fake_B/ 与 real_B/ 子目录（Pix2Pix 的 visualizer 默认产出）。
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from eval_core import run_evaluation

if __name__ == '__main__':
    run_evaluation('Pix2Pix')
