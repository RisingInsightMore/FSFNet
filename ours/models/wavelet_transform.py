# --*--coding   : utf-8  --*--
# @Time         : 2026/2/11 11:39
# @Author       : RisingInsight
# @File         : wavelet_transform2.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import torch
import torch.nn as nn
import numpy as np


class WaveletTransform2D(nn.Module):
    """单层Haar小波正变换（下采样+滤波），输出四个子带：LL, LH, HL, HH
       输入形状: (B, C, H, W)  输出形状: (B, C, H/2, W/2)
       修正：权重设置时使用 clone() 确保内存独立
    """
    def __init__(self, in_channels, gpu_ids):
        super().__init__()
        self.in_channels = in_channels
        self.gpu_ids = gpu_ids

        # Haar 滤波器 (归一化因子 1/√2)
        self.harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))   # [1/√2, 1/√2]
        self.harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        self.harr_wav_H[0, 0] = -1 * self.harr_wav_H[0, 0]   # [-1/√2, 1/√2]

        # 二维滤波器（外积）
        self.harr_wav_LL = np.transpose(self.harr_wav_L) * self.harr_wav_L   # [[0.5,0.5],[0.5,0.5]]
        self.harr_wav_LH = np.transpose(self.harr_wav_L) * self.harr_wav_H   # [[-0.5,0.5],[-0.5,0.5]]
        self.harr_wav_HL = np.transpose(self.harr_wav_H) * self.harr_wav_L   # [[-0.5,-0.5],[0.5,0.5]]
        self.harr_wav_HH = np.transpose(self.harr_wav_H) * self.harr_wav_H   # [[0.5,-0.5],[-0.5,0.5]]

        # 转换为 torch 张量，形状 (1, 2, 2)
        self.filter_LL = torch.from_numpy(self.harr_wav_LL).unsqueeze(0).cuda(self.gpu_ids)  # (1,2,2)
        self.filter_LH = torch.from_numpy(self.harr_wav_LH).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HL = torch.from_numpy(self.harr_wav_HL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HH = torch.from_numpy(self.harr_wav_HH).unsqueeze(0).cuda(self.gpu_ids)

        # 定义分组卷积层（每个输入通道独立处理）
        self.LL = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0,
                            bias=False, groups=in_channels).cuda(self.gpu_ids)
        self.LH = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0,
                            bias=False, groups=in_channels).cuda(self.gpu_ids)
        self.HL = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0,
                            bias=False, groups=in_channels).cuda(self.gpu_ids)
        self.HH = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0,
                            bias=False, groups=in_channels).cuda(self.gpu_ids)

        # 固定权重，不可训练，并使用 clone() 切断内存共享
        with torch.no_grad():
            for m in [self.LL, self.LH, self.HL, self.HH]:
                m.weight.requires_grad = False
            # 权重形状: (in_ch, 1, 2, 2)，先用 expand 复制，再 clone() 确保独立
            self.LL.weight.data = self.filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            self.LH.weight.data = self.filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            self.HL.weight.data = self.filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            self.HH.weight.data = self.filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    def forward(self, x):
        LL = self.LL(x)
        LH = self.LH(x)
        HL = self.HL(x)
        HH = self.HH(x)
        return LL, LH, HL, HH


class InverseWaveletTransform2D(nn.Module):
    """单层Haar小波逆变换（上采样+合成），输入四个子带，输出重建图像
       输入形状: 每个子带 (B, C, H/2, W/2)  输出形状: (B, C, H, W)
       修正：
         - 权重设置使用 clone() 避免内存共享
         - 逆变换滤波器系数与正变换相同（无需乘以2），以保证严格重建
       注：若正变换滤波器值为 0.5，逆变换同样使用 0.5 可完美重建。
         可以通过测试验证，若有偏差请自行调整。
    """
    def __init__(self, in_channels, gpu_ids):
        super().__init__()
        self.in_channels = in_channels
        self.gpu_ids = gpu_ids

        # 使用与正变换相同的滤波器
        self.harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        self.harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        self.harr_wav_H[0, 0] = -1 * self.harr_wav_H[0, 0]

        self.harr_wav_LL = np.transpose(self.harr_wav_L) * self.harr_wav_L
        self.harr_wav_LH = np.transpose(self.harr_wav_L) * self.harr_wav_H
        self.harr_wav_HL = np.transpose(self.harr_wav_H) * self.harr_wav_L
        self.harr_wav_HH = np.transpose(self.harr_wav_H) * self.harr_wav_H

        self.filter_LL = torch.from_numpy(self.harr_wav_LL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_LH = torch.from_numpy(self.harr_wav_LH).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HL = torch.from_numpy(self.harr_wav_HL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HH = torch.from_numpy(self.harr_wav_HH).unsqueeze(0).cuda(self.gpu_ids)

        # 转置卷积实现上采样+滤波，分组卷积
        self.LL_T = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2,
                                       bias=False, groups=in_channels).cuda(self.gpu_ids)
        self.LH_T = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2,
                                       bias=False, groups=in_channels).cuda(self.gpu_ids)
        self.HL_T = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2,
                                       bias=False, groups=in_channels).cuda(self.gpu_ids)
        self.HH_T = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2,
                                       bias=False, groups=in_channels).cuda(self.gpu_ids)

        # 固定权重，不可训练，使用 clone() 确保独立内存，系数与正变换相同
        with torch.no_grad():
            for m in [self.LL_T, self.LH_T, self.HL_T, self.HH_T]:
                m.weight.requires_grad = False
            # 权重设为滤波器值（无需乘以2），如果重建误差大，可尝试改为 2.0 * filter
            self.LL_T.weight.data = self.filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            self.LH_T.weight.data = self.filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            self.HL_T.weight.data = self.filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            self.HH_T.weight.data = self.filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    def forward(self, LL, LH, HL, HH):
        # 分别上采样后相加
        out = self.LL_T(LL) + self.LH_T(LH) + self.HL_T(HL) + self.HH_T(HH)
        return out


class MultiLevelWaveletTransform(nn.Module):
    """多层Haar小波正变换，递归分解低频子带，返回每层的子带列表
       参数：
           in_channels: 输入图像通道数
           gpu_ids: GPU 设备号
           levels: 分解层数
       输出：列表，每个元素为 (LL, LH, HL, HH) 元组，按分解顺序排列
    """
    def __init__(self, in_channels, gpu_ids, levels=2):
        super().__init__()
        self.levels = levels
        self.transforms = nn.ModuleList()
        for _ in range(levels):
            self.transforms.append(WaveletTransform2D(in_channels, gpu_ids))

    def forward(self, x):
        results = []
        current = x
        for t in self.transforms:
            ll, lh, hl, hh = t(current)
            results.append((ll, lh, hl, hh))
            current = ll   # 下一层继续分解低频子带
        return results