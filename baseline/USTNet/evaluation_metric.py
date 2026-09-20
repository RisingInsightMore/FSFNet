# --*--coding   : uft-8  --*--                         
# @Time         : 2025/11/17 20:32                 
# @Author       : RisingInsight                               
# @File         : evaluation_metric.py                     
# @Software     : PyCharm                 
# @Project      : USTNet-main                  
# @Function     :
import argparse
import os
import lpips
import numpy as np
from PIL import Image
from skimage import metrics
import torch
import time
import logging
from config import config

"""确保图像为RGB三通道"""
def ensure_rgb(image_tensor):
    if image_tensor.shape[1] == 1:                      # 灰度图转RGB
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
    return image_tensor


"""计算两幅图像的LPIPS指标"""
def calculate_lpips(img1, img2, opt, loss_fn):
    real_tensor = lpips.im2tensor(img1)                             # 转换为LPIPS张量
    fake_tensor = lpips.im2tensor(img2)

    if opt.use_gpu:
        real_tensor = real_tensor.cuda()                            # 转换为CUDA张量
        fake_tensor = fake_tensor.cuda()

    with torch.no_grad():
        distance = loss_fn.forward(real_tensor, fake_tensor)        # 计算LPIPS距离
    return distance.item()


"""计算两幅图像的SSIM指标"""
def calculate_ssim(img1, img2):
    img1 = np.array(img1).astype(np.float64)            # 转换为numpy数组并确保数据类型
    img2 = np.array(img2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")        # 检查图像尺寸是否一致

    if len(img1.shape) == 3 and img1.shape[2] == 3:     # RGB图像
        ssim_value = metrics.structural_similarity(img1, img2, channel_axis=-1, data_range=255)     # 计算SSIM
    else:                                               # 灰度图像
        ssim_value = metrics.structural_similarity(img1, img2, data_range=255)                      # 计算SSIM
    return ssim_value


"""计算两幅图像的RMSE指标"""
def calculate_rmse(img1, img2):
    img1 = np.array(img1).astype(np.float64)            # 转换为numpy数组并确保数据类型
    img2 = np.array(img2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")        # 检查图像尺寸是否一致

    mse = np.mean((img1 - img2) ** 2)           # 计算均方误差 (MSE)
    rmse = np.sqrt(mse)                         # 计算均方根误差 (RMSE)
    return rmse


"""计算两幅图像的PSNR指标, PSNR = 20*log10(MAX) - 10*log10(MSE)"""
def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float64)    # 转换为numpy数组并确保数据类型
    img2 = np.array(img2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")        # 检查图像尺寸是否一致

    mse = np.mean((img1 - img2) ** 2)           # 计算均方误差 (MSE)
    if mse == 0:
        return float('inf')                     # 如果MSE为0，说明图像完全相同，PSNR为无穷大

    max_pixel = 255.0                           # 像素最大值（8位图像为255）
    psnr = 20*np.log10(max_pixel) - 10*np.log10(mse)        # 计算PSNR
    # psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real', type=str, required=True, help='path to real infrared images')
    parser.add_argument('--fake', type=str, required=True, help='path to generated infrared images')
    parser.add_argument('--out', type=str, required=True, help='path to save metric results')
    parser.add_argument('-v', '--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use GPU for computation')

    opt = parser.parse_args()

    loss_fn = lpips.LPIPS(net='alex', version=opt.version)  # 初始化LPIPS模型
    if opt.use_gpu and torch.cuda.is_available():
        loss_fn.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        opt.use_gpu = False
        print("Using CPU")

    real_path = opt.real
    fake_path = opt.fake
    out_path = os.path.join(opt.out, 'results.txt')

    if not os.path.exists(real_path):
        raise ValueError(f"Real images path does not exist: {real_path}")           # 检查路径是否存在
    if not os.path.exists(fake_path):
        raise ValueError(f"Fake images path does not exist: {fake_path}")

    real_files = sorted([f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])        # 获取并排序文件列表，确保正确配对
    fake_files = sorted([f for f in os.listdir(fake_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(real_files) != len(fake_files):
        print(f"Warning: Number of images mismatch - Real: {len(real_files)}, Fake: {len(fake_files)}")             # 检查文件数量是否一致
        min_len = min(len(real_files), len(fake_files))                 # 取较小值进行计算
        real_files = real_files[:min_len]
        fake_files = fake_files[:min_len]
        print(f"Using first {min_len} images for calculation")
    if len(real_files) == 0:
        raise ValueError("No valid image files found in the specified directories")

    print(f"Found {len(real_files)} image pairs for metric calculation")

    dists_lpips, ssim_values, psnr_values, rmse_values = [], [], [], []
    successful_pairs = 0

    for real_file, fake_file in zip(real_files, fake_files):
        try:
            real_img_path = os.path.join(real_path, real_file)
            fake_img_path = os.path.join(fake_path, fake_file)

            dist_lpips = calculate_lpips(lpips.load_image(real_img_path), lpips.load_image(fake_img_path), opt, loss_fn)    # 计算LPIPS距离
            ssim_value = calculate_ssim(Image.open(real_img_path), Image.open(fake_img_path))                               # 计算SSIM
            psnr_value = calculate_psnr(Image.open(real_img_path), Image.open(fake_img_path))                               # 计算PSNR
            rmse_value = calculate_rmse(Image.open(real_img_path), Image.open(fake_img_path))                               # 计算RMSE

            successful_pairs += 1
            print(f'({real_file}, {fake_file}): LPIPS={dist_lpips:.6f}, SSIM={ssim_value:.6f}, PSNR={psnr_value:.6f}, RMSE={rmse_value:.6f}')
            dists_lpips.append(dist_lpips)
            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)
            rmse_values.append(rmse_value)

        except Exception as e:
            print(f"Error processing pair ({real_file}, {fake_file}): {str(e)}")
            continue

    if successful_pairs == 0:
        raise RuntimeError("No image pairs were successfully processed")

    # 统计结果
    dists_lpips_array, ssim_array, psnr_array, rmse_array = (np.array(dists_lpips), np.array(ssim_values),
                                                             np.array(psnr_values), np.array(rmse_values))
    avg_dist_lpips, avg_ssim, avg_psnr, avg_rmse = (np.mean(dists_lpips_array), np.mean(ssim_array),
                                                    np.mean(psnr_array), np.mean(rmse_array))
    std_dist_lpips, std_ssim, std_psnr, std_rmse = (np.std(dists_lpips_array), np.std(ssim_array),
                                                    np.std(psnr_array), np.std(rmse_array))
    min_dist_lpips, min_ssim, min_psnr, min_rmse = (np.min(dists_lpips_array), np.min(ssim_array),
                                                    np.min(psnr_array), np.min(rmse_array))
    max_dist_lpips, max_ssim, max_psnr, max_rmse = (np.max(dists_lpips_array), np.max(ssim_array),
                                                    np.max(psnr_array), np.max(rmse_array))

    with open(out_path, 'a') as f:
        f.write('=' * 60)
        f.write(f"\n{opt.out.split('/')[-2]} Metric Evaluation Results {config.timestamp}")
        f.write('\n' + '=' * 60)
        f.write(f"\nSuccessfully processed: {successful_pairs}/{len(real_files)} pairs")
        f.write(f"\nAverage LPIPS: {avg_dist_lpips:.6f} ± {std_dist_lpips:.6f}, Average SSIM: {avg_ssim:.6f} ± {std_ssim:.6f},"
                f" Average PSNR: {avg_psnr:.6f} ± {std_psnr:.6f}, Average RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
        f.write(f"\nMin LPIPS: {min_dist_lpips:.6f}, Min SSIM: {min_ssim:.6f}, Min PSNR: {min_psnr:.6f}, Min RMSE: {min_rmse:.6f}")
        f.write(f"\nMax LPIPS: {max_dist_lpips:.6f}, Max SSIM: {max_ssim:.6f}, Max PSNR: {max_psnr:.6f}, Max RMSE: {max_rmse:.6f}")
        f.write('\n' + '=' * 60 + '\n')
    print('\n' + '=' * 60)
    print('Metric Evaluation Results')
    print('=' * 60)
    print(f'Successfully processed: {successful_pairs}/{len(real_files)} pairs')
    print(f'Average LPIPS: {avg_dist_lpips:.6f} ± {std_dist_lpips:.6f}, Average SSIM: {avg_ssim:.6f} ± {std_ssim:.6f},'
          f' Average PSNR: {avg_psnr:.6f} ± {std_psnr:.6f}, Average RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}')
    print(f'Min LPIPS: {min_dist_lpips:.6f}, Min SSIM: {min_ssim:.6f}, Min PSNR: {min_psnr:.6f}, Min RMSE: {min_rmse:.6f}')
    print(f'Max LPIPS: {max_dist_lpips:.6f}, Max SSIM: {max_ssim:.6f}, Max PSNR: {max_psnr:.6f}, Max RMSE: {max_rmse:.6f}')
    print('=' * 60)
    # out_path = os.path.join(opt.out, 'log.txt')
    # with open(out_path, 'a') as f:
    #     f.write('\n' + '=' * 50)
    #     f.write(f"{(opt.out).split('/')[-2]} Metric Evaluation Results {config.timestamp}")
    #     f.write('\n' + '=' * 50)


if __name__ == '__main__':
    main()




