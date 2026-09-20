# ----coding   : utf-8  ----
# @Time         : 2026/5/15 11:31
# @Author       : RisingInsight
# @File         : networks6.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models.dynamic_parallel_window_attention import DynamicParallelWindowAttention
import functools
from timm.layers import trunc_normal_
import torch.nn.functional as F
from models.wavelet_transform import MultiLevelWaveletTransform, InverseWaveletTransform2D
from models.granular_ball import GranularBallModule

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
            self.granular_ball = GranularBallModule()
        self._granular_alpha = 0.0

        # ---------- Transformer 融合模块 ----------
        self.transformer_block = DynamicParallelWindowAttention(
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
    def __init__(self, in_channels=3, base_dim=192, levels=2):
        super().__init__()
        self.levels = levels
        self.level_encoders = nn.ModuleList()
        for _ in range(levels):
            self.level_encoders.append(nn.Sequential(
                nn.Conv2d(4 * in_channels, base_dim // 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(base_dim // 2),
                nn.LeakyReLU(),
                nn.Conv2d(base_dim // 2, base_dim // 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(base_dim // 2),
                nn.LeakyReLU()
            ))
        total_dim = (base_dim // 2) * levels
        self.low_fusion = nn.Sequential(
            nn.Conv2d(total_dim, base_dim, kernel_size=1),
            nn.InstanceNorm2d(base_dim),
            nn.LeakyReLU()
        )
        self.high_fusion = nn.Sequential(
            nn.Conv2d(total_dim, base_dim, kernel_size=1),
            nn.InstanceNorm2d(base_dim),
            nn.LeakyReLU()
        )
        self.downsample = nn.ModuleList()
        for i in range(3):
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(base_dim),
                    nn.LeakyReLU()
                )
            )

    def forward(self, wavelet_list):
        feats = []
        target_hw = wavelet_list[0][0].shape[2:]
        for lvl, (ll, lh, hl, hh) in enumerate(wavelet_list):
            cat = torch.cat([ll, lh, hl, hh], dim=1)
            feat = self.level_encoders[lvl](cat)
            if lvl > 0:
                feat = F.interpolate(feat, size=target_hw, mode='bilinear', align_corners=False)
            feats.append(feat)
        multi_feat = torch.cat(feats, dim=1)
        low = self.low_fusion(multi_feat)
        high = self.high_fusion(multi_feat)
        for down in self.downsample:
            low = down(low)
            high = down(high)
        return low, high


class FreqDecoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=3, gpu_ids=None):
        super().__init__()
        self.out_channels = out_channels
        self.gpu_ids = gpu_ids
        self.deconv1 = nn.ConvTranspose2d(in_channels, 192, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(192)
        self.lrelu1 = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(96)
        self.lrelu2 = nn.LeakyReLU()
        self.deconv3 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(48)
        self.lrelu3 = nn.LeakyReLU()

        self.to_LL = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)
        self.to_LH = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)
        self.to_HL = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)
        self.to_HH = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)

        self.inv_wavelet = InverseWaveletTransform2D(out_channels, gpu_ids)

    def forward(self, x):
        feat1 = self.lrelu1(self.norm1(self.deconv1(x)))
        feat2 = self.lrelu2(self.norm2(self.deconv2(feat1)))
        feat3 = self.lrelu3(self.norm3(self.deconv3(feat2)))

        LL = self.to_LL(feat3)
        LH = self.to_LH(feat3)
        HL = self.to_HL(feat3)
        HH = self.to_HH(feat3)

        out = self.inv_wavelet(LL, LH, HL, HH)
        out = torch.tanh(out)
        return out


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
