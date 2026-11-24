import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models.fgca import FrequencyGuidedTransformer
import functools
from timm.layers import trunc_normal_
import torch.nn.functional as F
from models.wavelet_transform import MultiLevelWaveletTransform, InverseWaveletTransform2D
from models.gagm import GranularityMap

class FSFNetModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # ---------- 空间编码器 ----------
        self.input_layer = nn.Sequential(
            nn.Conv2d(opt.input_nc, 48, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.enc_block_0 = UNetEncBlock(48, 48)
        self.enc_block_1 = UNetEncBlock(48, 96)
        self.enc_block_2 = UNetEncBlock(96, 192)
        self.enc_block_3 = UNetEncBlock(192, 384)

        # ---------- 频率编码器 ----------
        self.wavelet_transform = MultiLevelWaveletTransform(
            opt.input_nc, opt.gpu_ids[0], levels=2
        )
        self.freq_encoder = MultiScaleFreqEncoder(
            in_channels=opt.input_nc, base_dim=192, levels=2
        )

        # Granular ball module (optional)
        self.use_granular = getattr(opt, 'use_granular', False)
        if self.use_granular:
            self.granular_ball = GranularityMap()
        self._granular_alpha = 0.0

        # ---------- Transformer 融合模块 ----------
        self.transformer_block = FrequencyGuidedTransformer(
            depths=[12], num_heads=[12], embed_dim=384, window_size=4
        )

        # ---------- 多尺度频率融合模块（解码器各层） ----------
        self.freq_fuse_0 = nn.Sequential(  # 用于H/16 -> H/8
            nn.Conv2d(384 + 384, 384, kernel_size=1),
            nn.LeakyReLU()
        )
        self.freq_fuse_1 = nn.Sequential(  # 用于H/8 -> H/4
            nn.Conv2d(192 + 384, 192, kernel_size=1),
            nn.LeakyReLU()
        )
        self.freq_fuse_2 = nn.Sequential(  # 用于H/4 -> H/2
            nn.Conv2d(96 + 384, 96, kernel_size=1),
            nn.LeakyReLU()
        )
        self.freq_fuse_3 = nn.Sequential(  # 用于H/2 -> H
            nn.Conv2d(48 + 384, 48, kernel_size=1),
            nn.LeakyReLU()
        )

        # ---------- 空间解码器 ----------
        self.dec_block_0 = UNetDecBlock(384, 192)
        self.dec_block_1 = UNetDecBlock(192, 96)
        self.dec_block_2 = UNetDecBlock(96, 48)
        self.dec_block_3 = UNetDecBlock(48, 48)
        self.output_layer = nn.Sequential(
            nn.Conv2d(48, opt.output_nc, kernel_size=1),
            nn.Tanh()
        )

        # ---------- 频率解码器 ----------
        self.freq_decoder = FreqDecoder(
            in_channels=384, out_channels=opt.output_nc, gpu_ids=opt.gpu_ids[0]
        )

        # 可学习融合权重已移除，改为训练/测试分离返回

    def forward(self, x):
        # ---------- 空间编码 ----------
        x_input = self.input_layer(x)                       # [B,48, H, W]
        r0, y0 = self.enc_block_0(x_input)                  # r0:48,H,W; y0:48,H/2,W/2
        r1, y1 = self.enc_block_1(y0)                       # r1:96,H/2,W/2; y1:96,H/4,W/4
        r2, y2 = self.enc_block_2(y1)                       # r2:192,H/4,W/4; y2:192,H/8,W/8
        r3, y3 = self.enc_block_3(y2)                       # r3:384,H/8,W/8; y3:384,H/16,W/16

        # ---------- 频率编码 ----------
        wavelet_list = self.wavelet_transform(x)
        freq_low, freq_high = self.freq_encoder(wavelet_list)                   # [B,192, H/16, W/16]

        # Granularity map
        if self.use_granular:
            M_norm = self.granular_ball(y3)  # [B, 1, H_feat, W_feat]
        else:
            M_norm = None

        # ---------- Transformer 融合 ----------
        fused_feat = self.transformer_block(y3, freq_low, freq_high, M_norm, self._granular_alpha)  # [B,384, H/16, W/16]

        # ---------- 多尺度频率引导解码 ----------
        # 逐级上采样频率特征并融合
        # 第0层 (H/16 -> H/8)
        freq_low_up = F.interpolate(freq_low, scale_factor=2, mode='bilinear')
        freq_high_up = F.interpolate(freq_high, scale_factor=2, mode='bilinear')
        freq_cat = torch.cat([freq_low_up, freq_high_up], dim=1)                # [B,384, H/8, W/8]
        fused_up = F.interpolate(fused_feat, scale_factor=2, mode='bilinear')
        fused_fused = self.freq_fuse_0(torch.cat([fused_up, freq_cat], dim=1))  # [B,384, H/8, W/8]
        x_dec0 = self.dec_block_0(fused_fused, r3)                                      # [B,192, H/8, W/8]

        # 第1层 (H/8 -> H/4)
        freq_low_up = F.interpolate(freq_low_up, scale_factor=2, mode='bilinear')
        freq_high_up = F.interpolate(freq_high_up, scale_factor=2, mode='bilinear')
        freq_cat = torch.cat([freq_low_up, freq_high_up], dim=1)                 # [B,384, H/4, W/4]
        x_up = F.interpolate(x_dec0, scale_factor=2, mode='bilinear')
        x_fused = self.freq_fuse_1(torch.cat([x_up, freq_cat], dim=1))           # [B,192, H/4, W/4]
        x_dec1 = self.dec_block_1(x_fused, r2)                                           # [B,96,  H/4, W/4]

        # 第2层 (H/4 -> H/2)
        freq_low_up = F.interpolate(freq_low_up, scale_factor=2, mode='bilinear')
        freq_high_up = F.interpolate(freq_high_up, scale_factor=2, mode='bilinear')
        freq_cat = torch.cat([freq_low_up, freq_high_up], dim=1)  # [B,384, H/2, W/2]
        x_up = F.interpolate(x_dec1, scale_factor=2, mode='bilinear')
        x_fused = self.freq_fuse_2(torch.cat([x_up, freq_cat], dim=1))  # [B,96, H/2, W/2]
        x_dec2 = self.dec_block_2(x_fused, r1)                # [B,48,  H/2, W/2]

        # 第3层 (H/2 -> H)
        freq_low_up = F.interpolate(freq_low_up, scale_factor=2, mode='bilinear')
        freq_high_up = F.interpolate(freq_high_up, scale_factor=2, mode='bilinear')
        freq_cat = torch.cat([freq_low_up, freq_high_up], dim=1)  # [B,384, H, W]
        x_up = F.interpolate(x_dec2, scale_factor=2, mode='bilinear')
        x_fused = self.freq_fuse_3(torch.cat([x_up, freq_cat], dim=1))  # [B,48, H, W]
        x_dec3 = self.dec_block_3(x_fused, r0)                # [B,48,  H,   W]
        spatial_out = self.output_layer(x_dec3)              # [B,3,   H,   W]

        # ---------- 频率解码 ----------
        freq_out = self.freq_decoder(fused_feat)             # [B,3,   H,   W]

        # ---------- 根据训练/测试模式返回 ----------
        if self.training:
            return spatial_out, freq_out   # 训练时返回双输出
        else:
            return spatial_out              # 测试时只返回清晰的空间输出

    def set_granular_alpha(self, alpha):
        self._granular_alpha = alpha


class UNetBasicBlock(nn.Module):
    def __init__(self, in_feature, out_feature, mid_feature=None):
        super().__init__()
        if mid_feature is None:
            mid_feature = out_feature
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_feature),
            nn.Conv2d(in_feature, mid_feature, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(mid_feature),
            nn.Conv2d(mid_feature, out_feature, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNetEncBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.block = UNetBasicBlock(in_feature, out_feature)
        self.downsample = nn.Conv2d(out_feature, out_feature, kernel_size=2, stride=2)

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return r, y


class UNetDecBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_feature, in_feature, kernel_size=3, padding=1),
        )
        self.block = UNetBasicBlock(in_feature * 2, out_feature, mid_feature=in_feature)

    def forward(self, x, r):
        # x = self.upsample(x)
        y = torch.cat([x, r], dim=1)
        y = self.block(y)
        return y


class MultiScaleFreqEncoder(nn.Module):
    """Orthogonal Haar wavelet frequency encoder E_f (Section 3.2.2 of the paper).

    The n-level Haar decomposition provides 4n subbands (LL_j, LH_j, HL_j, HH_j),
    j = 1, ..., n. Each subband is encoded by an independent convolutional branch,
    and all subband features are resized to H/16 x W/16 by bilinear interpolation.
    A 1x1 convolution over the LL features of all levels gives the low-frequency
    representation F_low, and a 1x1 convolution over the LH, HL, and HH features of
    all levels gives the high-frequency representation F_high.

    Args:
        in_channels (int): channels of each subband (3 for RGB inputs).
        base_dim (int): channels of F_low and F_high (192).
        levels (int): number of decomposition levels n (2).
        branch_dim (int): channels of each subband branch.

    Inputs:
        wavelet_list (list): output of ``MultiLevelWaveletTransform``, one
            (LL, LH, HL, HH) tuple per level; level j has resolution H/2^j x W/2^j.

    Returns:
        tuple[Tensor, Tensor]: F_low and F_high, each of shape (B, 192, H/16, W/16).
    """

    def __init__(self, in_channels=3, base_dim=192, levels=2, branch_dim=120):
        super().__init__()
        self.levels = levels
        # one independent branch per subband: 4 subbands x n levels
        self.branches = nn.ModuleList([
            nn.ModuleList([self._make_branch(in_channels, branch_dim) for _ in range(4)])
            for _ in range(levels)
        ])
        self.low_proj = nn.Sequential(
            nn.Conv2d(branch_dim * levels, base_dim, kernel_size=1),
            nn.InstanceNorm2d(base_dim),
            nn.LeakyReLU()
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(3 * branch_dim * levels, base_dim, kernel_size=1),
            nn.InstanceNorm2d(base_dim),
            nn.LeakyReLU()
        )

    @staticmethod
    def _make_branch(in_channels, dim):
        return nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU()
        )

    def forward(self, wavelet_list):
        h1, w1 = wavelet_list[0][0].shape[2:]            # level-1 subbands: H/2 x W/2
        target_hw = (h1 // 8, w1 // 8)                   # H/16 x W/16
        low_feats, high_feats = [], []
        for lvl, subbands in enumerate(wavelet_list):    # subbands = (LL, LH, HL, HH)
            feats = [F.interpolate(self.branches[lvl][k](band), size=target_hw,
                                   mode='bilinear', align_corners=False)
                     for k, band in enumerate(subbands)]
            low_feats.append(feats[0])                   # LL_j
            high_feats.extend(feats[1:])                 # LH_j, HL_j, HH_j
        freq_low = self.low_proj(torch.cat(low_feats, dim=1))
        freq_high = self.high_proj(torch.cat(high_feats, dim=1))
        return freq_low, freq_high


class FreqDecoder(nn.Module):
    """Training-only auxiliary wavelet decoder D_f (Section 3.5 of the paper).

    The implementation of this module will be released upon acceptance of the paper.

    D_f takes the fused bottleneck feature F_fused of the infrared pass, predicts the
    four Haar subbands (LL, LH, HL, HH) at H/2 x W/2, and assembles them into a
    full-resolution image with the fixed inverse wavelet transform. It is used only
    by the auxiliary loss L_aux during translation training and is removed at
    inference.

    Inputs:
        x (Tensor): fused bottleneck feature F_fused, shape (B, 384, H/16, W/16).

    Returns:
        Tensor: reconstructed infrared image y_freq, shape (B, 3, H, W).
    """

    def __init__(self, in_channels=384, out_channels=3, gpu_ids=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        raise NotImplementedError(
            "The auxiliary wavelet decoder will be released upon acceptance of the paper.")


class NLayerDiscriminator(nn.Module):
    def __init__(self, opt, ndf=64, n_layers=3, max_mult=8):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(opt.input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, max_mult)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, max_mult)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def linear_scheduler(optimizer, epochs_warmup, epochs_anneal):
    def lambda_rule(epoch, warmup, anneal):
        if epoch < warmup:
            return 1.0
        return 1.0 - (epoch - warmup) / (anneal + 1)
    lr_fn = lambda epoch: lambda_rule(epoch, epochs_warmup, epochs_anneal)
    return lr_scheduler.LambdaLR(optimizer, lr_fn)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        return linear_scheduler(optimizer, opt.epochs_warmup, opt.epochs_anneal)
    return None
