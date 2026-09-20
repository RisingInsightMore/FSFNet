import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models.dynamic_parallel_window_attention import DynamicParallelWindowAttention
import functools
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_

class USTNetModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(
            opt.input_nc, 48,
            kernel_size=3, padding=1),
            nn.LeakyReLU()
        )       # 卷积核大小为3，步长为1，填充为1的卷积层

        self.enc_block_0 = UNetEncBlock(48, 48)     # 编码器块，编码器块，先两个卷积层，后下采样。提取特征
        self.enc_block_1 = UNetEncBlock(48, 96)
        self.enc_block_2 = UNetEncBlock(96, 192)
        self.enc_block_3 = UNetEncBlock(192, 384)

        self.transformer_block = DynamicParallelWindowAttention()      #

        self.dec_block_0 = UNetDecBlock(384, 192)   # 解码器块，先上采样，残差连接，再卷积层。高分辨率重建
        self.dec_block_1 = UNetDecBlock(192, 96)
        self.dec_block_2 = UNetDecBlock(96, 48)
        self.dec_block_3 = UNetDecBlock(48, 48)

        self.output_layer = nn.Sequential(nn.Conv2d(
            48, opt.output_nc,
            kernel_size=1), nn.Tanh(),
        )       # 卷积核大小为1，步长为1，填充为0的卷积层

    def forward(self, x):
        x_input = self.input_layer(x)                       # [nb, input_nc, h, w] -> [nb, 48, h, w]
        x_enc_r_0, x_enc_y_0 = self.enc_block_0(x_input)    # [nb, 48, h, w]       -> [nb, 48, h, w]      [nb, 48, h/2, w/2]
        x_enc_r_1, x_enc_y_1 = self.enc_block_1(x_enc_y_0)  # [nb, 48, h/2, w/2]   -> [nb, 96, h/2, w/2]  [nb, 96, h/4, w/4]
        x_enc_r_2, x_enc_y_2 = self.enc_block_2(x_enc_y_1)  # [nb, 96, h/4, w/4]   -> [nb, 192, h/4, w/4] [nb,192, h/8, w/8]
        x_enc_r_3, x_enc_y_3 = self.enc_block_3(x_enc_y_2)  # [nb,192, h/8, w/8]   -> [nb, 384, h/8, w/8] [nb,384, h/16, w/16]

        x_inner = self.transformer_block(x_enc_y_3)         # [nb,384, h/16, w/16] -> [nb,384, h/16, w/16]

        x_dec_y_0 = self.dec_block_0(x_inner, x_enc_r_3)    # [nb,384, h/16, w/16] [nb, 384, h/8, w/8] -> [nb, 192, h/8, w/8]
        x_dec_y_1 = self.dec_block_1(x_dec_y_0, x_enc_r_2)  # [nb, 192, h/8, w/8]  [nb, 192, h/4, w/4] -> [nb, 96, h/4, w/4]
        x_dec_y_2 = self.dec_block_2(x_dec_y_1, x_enc_r_1)  # [nb, 96, h/4, w/4]   [nb, 96, h/2, w/2]  -> [nb, 48, h/2, w/2]
        x_dec_y_3 = self.dec_block_3(x_dec_y_2, x_enc_r_0)  # [nb, 48, h/2, w/2]   [nb, 48, h, w]      -> [nb, 48, h, w]

        x_output = self.output_layer(x_dec_y_3)             # [nb, 48, h, w] -> [nb, output_nc, h, w]

        return x_output

"""
    两个卷积层，提取特征，保证分辨率，输入和输出的图像大小不变
    [nb, in_feature, h, w] -> [nb, mid_feature, (h-3+2*1)/1+1, (w-3+2*1)/1+1] -> [nb, out_feature, (h-3+2*1)/1+1, (w-3+2*1)/1+1]
"""
class UnetBasicBlock(nn.Module):
    def __init__(self, in_feature, out_feature, mid_feature=None):
        super().__init__()
        if mid_feature is None:
            mid_feature = out_feature
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_feature),                                      # 实例归一化，对每个样本（实例）的特征进行归一化
            nn.Conv2d(in_feature, mid_feature, kernel_size=3, padding=1),       # 卷积层，卷积核大小为3，步长1，填充1
            nn.LeakyReLU(),                                                     # 激活函数
            nn.InstanceNorm2d(mid_feature),                                     # 实例归一化
            nn.Conv2d(mid_feature, out_feature, kernel_size=3, padding=1),      # 卷积层
            nn.LeakyReLU()                                                      # 激活函数
        )

    def forward(self, x):
        return self.block(x)    # [nb, in_feature, h, w] -> [nb, out_feature, h, w]


"""
    编码器块，先两个卷积层，后下采样。提取特征
    [nb, in_feature, h, w] -> [nb, out_feature, h, w] -> [nb, out_feature, (h-2+2*0)/2+1, (w-2+2*0)/2+1]
"""
class UNetEncBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.block = UnetBasicBlock(in_feature, out_feature, mid_feature=None)          # 两个卷积层
        self.downsample = nn.Conv2d(out_feature, out_feature, kernel_size=2, stride=2)  # 下采样，图像长宽都变为原来一半，卷积核大小为2，步长为2，填充为0

    def forward(self, x):
        r = self.block(x)           # [nb, in_feature, h, w] -> [nb, out_feature, h, w]
        y = self.downsample(r)      # [nb, out_feature, h, w] -> [nb, out_feature, h/2, w/2]
        return r, y


"""
    解码器块，先上采样，残差连接，再卷积层。高分辨率重建
    [nb, in_feature, h, w] -> [nb, in_feature, 2*h, 2*w] -> [nb, in_feature+r_channels, 2*h, 2*w] -> [nb, out_feature, 2*h, 2*w]
"""
class UNetDecBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),                                        # 上采样层，图像长宽都变为原来2倍
            nn.Conv2d(in_feature, in_feature, kernel_size=3, padding=1),        # 卷积层，卷积核大小为3，步长1，填充1
        )
        self.block = UnetBasicBlock(in_feature * 2, out_feature, mid_feature=in_feature)    # 两个卷积层

    def forward(self, x, r):
        x = self.upsample(x)                    # [nb, in_feature, h, w] -> [nb, in_feature, 2*h, 2*w]
        y = torch.cat([x, r], dim=1)    # 残差连接，[nb, in_feature, 2*h, 2*w] -> [nb, in_feature+r_channels, 2*h, 2*w]
        y = self.block(y)                       # [nb, in_feature+r_channels, 2*h, 2*w] -> [nb, out_feature, 2*h, 2*w]
        return y


"""
    判别器输出的是一个二维的特征图，每个元素对应输入图像的一个感受野（patch），因此被称为PatchGAN
    PatchGAN判别器：输出一个特征图而不是单个值，每个位置对应输入图像的一个patch的真伪判断
    [nb, input_nc, H, W] -> [nb, 64, H/2, W/2] -> [nb, 128, H/4, W/4] -> ... -> [nb, max_mult*ndf, H/(2^n_layers), W/(2^n_layers)] -> [nb, 1, H/(2^n_layers), W/(2^n_layers)]
"""
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, opt, ndf=64, n_layers=3, max_mult=8):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        # 直接使用类
        norm_layer = nn.BatchNorm2d                             # BatchNorm2d 作为归一化
        # 使用partial预设参数
        # norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)

        if type(norm_layer) == functools.partial:               # 若norm_layer是通过partial创建的包装函数
            use_bias = norm_layer.func == nn.InstanceNorm2d     # 若为InstanceNorm2d，use_bias为True, 卷积层不需要偏置（InstanceNorm2d 内部已经包含了偏置项）
        else:
            use_bias = norm_layer == nn.InstanceNorm2d          # 若为BatchNorm2d，use_bias为False，卷积层需要偏置（BatchNorm2d 不包含偏置项）

        kw = 4                  # 卷积核大小为4x4
        padw = 1                # 填充大小为1

        # 第一层积积+激活函数，进行下采样。[nb, input_nc, H, W] -> [nb, 64, H/2, W/2]
        sequence = [nn.Conv2d(opt.input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]     # LeakyReLU负斜率为0.2，inplace=True节省内存

        nf_mult = 1                     # 当前通道倍数
        nf_mult_prev = 1                # 前一层通道倍数

        # 构建中间卷积层：逐渐增加通道数，减少空间尺寸，进行下采样
        for n in range(1, n_layers):                # 从第1层到第n_layers-1层
            nf_mult_prev = nf_mult                  # 保存当前倍数作为下一层的输入倍数
            nf_mult = min(2 ** n, max_mult)         # 计算新的倍数2^n，但不超过max_mult限制

            # 添加卷积层 + 归一化 + 激活函数。 [nb 64*nf_mult_prev, H/(2^n), W/(2^n)] -> [nb, 64*nf_mult, H/(2^(n+1)), W/(2^(n+1))]
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),                          # 批归一化
                nn.LeakyReLU(0.2, True)         # 激活函数
            ]

        # 倒数第二层：保持空间尺寸，可能继续增加通道数，不超过max_mult限制
        nf_mult_prev = nf_mult                      # 保存倒数第三层输入倍数作为倒数第二层的输入倍数
        nf_mult = min(2 ** n_layers, max_mult)      # 计算新的倍数2^n，但不超过max_mult限制

        # 倒数第二层stride=1，不进行下采样。[nb, 64*nf_mult_prev, H/(2^n_layers), W/(2^n_layers)] -> [nb, 64*nf_mult, H/(2^n_layers), W/(2^n_layers))
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 最后一层是输出层，将通道数降为1（真/假判断）。【nb, 64*nf_mult, H/(2^n_layers), W/(2^n_layers)] -> [nb, 1, H/(2^n_layers), W/(2^n_layers)]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)            # [nb, input_nc, H, W] -> [nb, 1,  H/(2^n_layers), W/(2^n_layers)]


def linear_scheduler(optimizer, epochs_warmup, epochs_anneal):
    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:           # 预训练阶段，学习率线性增加
            return 1.0

        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)      # 线性衰减阶段

    # 刚开始时epoch为0，学习率为0，随着epoch增加，学习率线性增加，直到达到最大学习率，然后线性衰减
    lr_fn = lambda epoch: lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        return linear_scheduler(optimizer, opt.epochs_warmup, opt.epochs_anneal)       # 线性学习率调度器
