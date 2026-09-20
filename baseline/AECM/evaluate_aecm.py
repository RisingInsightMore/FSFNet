# evaluate_aecm.py
# 评估 AECM 推理结果，输出 6 个指标：FID↓ KID↓ LPIPS↓ RMSE↓ SSIM↑ PSNR↑
#
# 与仓库 improved/evaluation_metric.py 的关系（目标：严格可比）:
#   - LPIPS 计算与 improved 版【逐字一致】：lpips.im2tensor(lpips.load_image(path))，
#     即保留 load_image 已返回 [-1,1] 张量后再 im2tensor 的写法，确保与
#     USTNet(improved) 的 LPIPS 数值口径完全相同、可直接进同一张论文对比表。
#     （注意：improved 该写法非“标准”LPIPS，但为论文各模型口径统一，刻意保持一致。）
#   - SSIM / PSNR / RMSE 实现与 improved 版一致（data_range=255 等）。
#   - FID / KID 通过 torch_fidelity 计算，与 fidelity CLI（同引擎、同默认 seed=0）等价。
#   其余为健壮性改进：不依赖 config、按文件名精确配对、兼容子目录布局、精度控制。
#
# 用法示例:
#   python evaluate_aecm.py --input ./results/AECM_AVIID/test_400/images \
#                           --out ./eval_AECM_AVIID --fid
import argparse
import os
import sys
import shutil
import datetime
import numpy as np
from PIL import Image
import torch


def _key(stem):
    """统一配对 key：无论文件名含不含 _fake_B/_real_B 后缀都能对齐。"""
    return stem.replace('_fake_B', '').replace('_real_B', '')


def find_pairs(images_dir):
    """兼容扁平 / 子目录两种布局，返回 (pairs, n_real, n_fake)。
    pairs 每个元素是 (real_path, fake_path) 完整路径。"""
    sub_fake = os.path.join(images_dir, 'fake_B')
    sub_real = os.path.join(images_dir, 'real_B')
    if os.path.isdir(sub_fake) and os.path.isdir(sub_real):
        fake_dir, real_dir = sub_fake, sub_real
        print('[layout] detected SUBDIR mode: images/fake_B + images/real_B')
    else:
        fake_dir = real_dir = images_dir
        print('[layout] detected FLAT mode: images/*_fake_B / *_real_B')

    def _collect(d):
        m = {}
        for f in os.listdir(d):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                m[_key(os.path.splitext(f)[0])] = f
        return m

    fake_map, real_map = _collect(fake_dir), _collect(real_dir)
    common = sorted(set(fake_map) & set(real_map))
    pairs = [(os.path.join(real_dir, real_map[b]),
              os.path.join(fake_dir, fake_map[b])) for b in common]
    return pairs, len(real_map), len(fake_map)


def load_np(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float64)


def calc_ssim(img1, img2):
    from skimage import metrics
    if img1.shape != img2.shape:
        raise ValueError('shape mismatch %s vs %s' % (img1.shape, img2.shape))
    try:
        return metrics.structural_similarity(img1, img2, channel_axis=-1, data_range=255)
    except TypeError:
        return metrics.structural_similarity(img1, img2, multichannel=True, data_range=255)


def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def calc_rmse(img1, img2):
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))


def stat(x):
    a = np.array(x, dtype=np.float64)
    return a.mean(), a.std(), a.min(), a.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help='AECM results images dir, e.g. ./results/AECM_AVIID/test_400/images')
    parser.add_argument('--real', type=str, default=None, help='override: dir of real_B images')
    parser.add_argument('--fake', type=str, default=None, help='override: dir of fake_B images')
    parser.add_argument('--out', type=str, required=True, help='dir to save results.txt')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use GPU if available')
    parser.add_argument('--fid', action='store_true',
                        help='also compute FID/KID (needs torch_fidelity installed)')
    opt = parser.parse_args()

    use_gpu = opt.use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    import lpips
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    if use_gpu:
        loss_fn = loss_fn.to(device)
        print('LPIPS running on GPU:', torch.cuda.get_device_name(0))
    loss_fn.eval()

    # ---- 定位 real / fake 配对 ----
    if opt.real and opt.fake:
        real_dir, fake_dir = opt.real, opt.fake
        real_files = [f for f in os.listdir(real_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        fake_files = [f for f in os.listdir(fake_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        real_map = {_key(os.path.splitext(f)[0]): f for f in real_files}
        fake_map = {_key(os.path.splitext(f)[0]): f for f in fake_files}
        common = sorted(set(real_map) & set(fake_map))
        pairs = [(os.path.join(real_dir, real_map[b]),
                  os.path.join(fake_dir, fake_map[b])) for b in common]
        n_real, n_fake = len(real_map), len(fake_map)
        print('Found %d paired images (real_B=%d, fake_B=%d in dir)'
              % (len(pairs), n_real, n_fake))
    elif opt.input:
        pairs, n_real, n_fake = find_pairs(opt.input)
        if len(pairs) == 0:
            sys.exit('ERROR: No real_B / fake_B pairs found in %s\n'
                     '       请确认该目录下存在 fake_B/ 与 real_B/ 子目录，'
                     '或存在 *_fake_B.* / *_real_B.* 文件。' % opt.input)
        print('Found %d paired images (real_B=%d, fake_B=%d in dir)'
              % (len(pairs), n_real, n_fake))
    else:
        sys.exit('ERROR: Provide either --input or both --real and --fake')

    os.makedirs(opt.out, exist_ok=True)

    # ---- 逐图像素级指标 ----
    lpips_vals, ssim_vals, psnr_vals, rmse_vals = [], [], [], []
    for rp, fp in pairs:
        # LPIPS: 与 improved/evaluation_metric.py 严格一致 —— im2tensor(load_image(...))
        with torch.no_grad():
            d = loss_fn(lpips.im2tensor(lpips.load_image(rp)).to(device),
                        lpips.im2tensor(lpips.load_image(fp)).to(device))
        lpips_vals.append(d.item())
        r64, f64 = load_np(rp), load_np(fp)
        ssim_vals.append(calc_ssim(r64, f64))
        psnr_vals.append(calc_psnr(r64, f64))
        rmse_vals.append(calc_rmse(r64, f64))

    lpips_mean, lpips_std, _, _ = stat(lpips_vals)
    ssim_mean, ssim_std, _, _ = stat(ssim_vals)
    psnr_mean, psnr_std, _, _ = stat(psnr_vals)
    rmse_mean, rmse_std, _, _ = stat(rmse_vals)

    # ---- FID / KID（可选，torch_fidelity）----
    fid_val = kid_mean = kid_std = None
    if opt.fid:
        try:
            from torch_fidelity import calculate_metrics
        except ImportError:
            print('WARNING: torch_fidelity not installed -> skip FID/KID. '
                  'Install: pip install torch-fidelity')
        else:
            stage_real = os.path.join(opt.out, 'fid_real')
            stage_fake = os.path.join(opt.out, 'fid_fake')
            os.makedirs(stage_real, exist_ok=True)
            os.makedirs(stage_fake, exist_ok=True)
            for rp, fp in pairs:
                shutil.copy(rp, os.path.join(stage_real, os.path.basename(rp)))
                shutil.copy(fp, os.path.join(stage_fake, os.path.basename(fp)))
            m = calculate_metrics(input1=stage_real, input2=stage_fake,
                                  cuda=use_gpu, fid=True, kid=True,
                                  kid_subset_size=min(500, len(pairs)), verbose=False)
            fid_val = m['frechet_inception_distance']
            kid_mean = m['kernel_inception_distance_mean']
            kid_std = m['kernel_inception_distance_std']

    # ---- 汇总 ----
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    lines = []
    lines.append('=' * 64)
    lines.append('AECM Evaluation  %s' % tag)
    lines.append('=' * 64)
    lines.append('Paired images : %d' % len(pairs))
    # 精度约定：FID 保留 5 位；其余指标（KID/LPIPS/SSIM/PSNR/RMSE）严格 6 位
    lines.append('LPIPS : %.6f +- %.6f   (lower is better)' % (lpips_mean, lpips_std))
    lines.append('SSIM  : %.6f +- %.6f   (higher is better)' % (ssim_mean, ssim_std))
    lines.append('PSNR  : %.6f +- %.6f   (higher is better)' % (psnr_mean, psnr_std))
    lines.append('RMSE  : %.6f +- %.6f   (lower is better)' % (rmse_mean, rmse_std))
    if fid_val is not None:
        lines.append('FID   : %.5f         (lower is better)' % fid_val)
        lines.append('KID   : %.6f +- %.6f   (lower is better)' % (kid_mean, kid_std))
    lines.append('=' * 64)
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(opt.out, 'results.txt'), 'a') as f:
        f.write(txt + '\n')

    if fid_val is None:
        print('\nFID/KID 未在此步计算。获取方式二选一:')
        print('  (A) 重跑加 --fid（需已装 torch_fidelity）')
        print('  (B) 用仓库的 fidelity CLI（需先 split_sample 拆出 images_fake/images_real）')


if __name__ == '__main__':
    main()
