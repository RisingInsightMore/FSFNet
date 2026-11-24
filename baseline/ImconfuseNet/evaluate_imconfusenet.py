# evaluate_imconfusenet.py
# 统一红外(可见光->红外)翻译指标评估入口（与 AECM 口径严格一致）。
# 用法:
#   cd ./baseline/ImconfuseNet
#   python evaluate_imconfusenet.py --input ./results/ImconfuseNet_AVIID/test_400/images \
#                                   --out ./eval_ImconfuseNet_AVIID --fid
# 说明: 默认算 LPIPS/SSIM/PSNR/RMSE；加 --fid 额外算 FID/KID。
#       依赖 results 目录下存在 fake_B/(或 fake_B_infrared/) 与 real_B/ 子目录
#       （ImconfuseNet 的 visualizer 默认产出；引擎会自动识别 *_infrared 变体）。
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from eval_core import run_evaluation

if __name__ == '__main__':
    run_evaluation('ImconfuseNet')
