# # --*--coding   : uft-8  --*--
# # @Time         : 2025/10/22 23:53
# # @Author       : RisingInsight
# # @File         : wavelet_transform_Dual_Tree_Comple.py
# # @Software     : PyCharm
# # @Project      : FSFNet-main
# # @Function     :
# import torch
# import torch.nn as nn
# from numpy import ndarray, sqrt
#
# from pytorch_wavelets.dtcwt.coeffs import qshift as _qshift, biort as _biort, level1
# from pytorch_wavelets.dtcwt.lowlevel import prep_filt
# from pytorch_wavelets.dtcwt.transform_funcs import FWD_J1, FWD_J2PLUS
# from pytorch_wavelets.dtcwt.transform_funcs import INV_J1, INV_J2PLUS
# from pytorch_wavelets.dtcwt.transform_funcs import get_dimensions6
# from pytorch_wavelets.dwt.lowlevel import mode_to_int
# from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse
#
#
# def pm(a, b):
#     u = (a + b)/sqrt(2)     # 实部
#     v = (a - b)/sqrt(2)     # 虚部
#     return u, v
#
#
# class DTCWTForward(nn.Module):
#     def __init__(self, biort='near_sym_a', qshift='qshift_a',
#                  J=3, skip_hps=False, include_scale=False,
#                  o_dim=2, ri_dim=-1, mode='symmetric'):
#         super().__init__()
#         if o_dim == ri_dim:
#             raise ValueError("Orientations and real/imaginary parts must be "
#                              "in different dimensions.")
#
#         self.biort = biort          # 第一级小波基，用于分解低频分量，双正交
#         self.qshift = qshift        # 第二级及以后分解的滤波器，用于分解高频分量，q-shift滤波器
#         self.J = J                  # 分解层数
#         self.o_dim = o_dim          # 方向维数，默认2，即6个方向
#         self.ri_dim = ri_dim        # 实部和虚部的维度，默认-1，即最后一维
#         self.mode = mode            # 边界处理模式，默认对称扩展
#         if isinstance(biort, str):
#             h0o, _, h1o, _ = _biort(biort)      # 第一级小波基的低通和高通滤波器
#             self.register_buffer('h0o', prep_filt(h0o, 1))
#             self.register_buffer('h1o', prep_filt(h1o, 1))
#         else:
#             self.register_buffer('h0o', prep_filt(biort[0], 1))
#             self.register_buffer('h1o', prep_filt(biort[1], 1))
#         if isinstance(qshift, str):
#             # 第二级及后后分解的高通滤波器，h0a和h0b是树A和树B低通滤波器，h1a和h1b是树A和树B高通滤波器
#             h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
#             self.register_buffer('h0a', prep_filt(h0a, 1))
#             self.register_buffer('h0b', prep_filt(h0b, 1))
#             self.register_buffer('h1a', prep_filt(h1a, 1))
#             self.register_buffer('h1b', prep_filt(h1b, 1))
#         else:
#             self.register_buffer('h0a', prep_filt(qshift[0], 1))
#             self.register_buffer('h0b', prep_filt(qshift[1], 1))
#             self.register_buffer('h1a', prep_filt(qshift[2], 1))
#             self.register_buffer('h1b', prep_filt(qshift[3], 1))
#
#         # Get the function to do the DTCWT
#         if isinstance(skip_hps, (list, tuple, ndarray)):
#             self.skip_hps = skip_hps
#         else:
#             self.skip_hps = [skip_hps,] * self.J
#         if isinstance(include_scale, (list, tuple, ndarray)):
#             self.include_scale = include_scale
#         else:
#             self.include_scale = [include_scale,] * self.J
#
#     def forward(self, x):
#         scales = [x.new_zeros([]),] * self.J        # 各层低频系数
#         highs = [x.new_zeros([]),] * self.J         # 各层高频系数
#         mode = mode_to_int(self.mode)
#         if self.J == 0:
#             return x, None
#
#         # 确保输入图像的高宽是2的整数倍
#         r, c = x.shape[2:]
#         if r % 2 != 0:
#             x = torch.cat((x, x[:,:,-1:]), dim=2)       # 行扩展
#         if c % 2 != 0:
#             x = torch.cat((x, x[:,:,:,-1:]), dim=3)     # 列扩展
#
#         # 使用biort滤波器进行第一级分解
#         low, h = FWD_J1.apply(x, self.h0o, self.h1o, self.skip_hps[0],
#                               self.o_dim, self.ri_dim, mode)
#         highs[0] = h        # 第一级高频系数（6个方向）
#         if self.include_scale[0]:
#             scales[0] = low
#
#         for j in range(1, self.J):
#             # 确保低频系数是4的整数倍
#             r, c = low.shape[2:]
#             if r % 4 != 0:
#                 low = torch.cat((low[:,:,0:1], low, low[:,:,-1:]), dim=2)
#             if c % 4 != 0:
#                 low = torch.cat((low[:,:,:,0:1], low, low[:,:,:,-1:]), dim=3)
#
#             # 使用qshift滤波器进行后续分解
#             low, h = FWD_J2PLUS.apply(low, self.h0a, self.h1a, self.h0b,
#                                       self.h1b, self.skip_hps[j], self.o_dim,
#                                       self.ri_dim, mode)
#             highs[j] = h        # 后续高频系数（6个方向）
#             if self.include_scale[j]:
#                 scales[j] = low
#
#         if True in self.include_scale:
#             return scales, highs
#         else:
#             return low, highs


import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class DTCWTTransform2D(nn.Module):
    def __init__(self, J=2, biort='near_sym_a', qshift='qshift_a', include_scale=False, gpu_ids=None):
        super().__init__()
        self.J = J
        self.include_scale = include_scale
        self.gpu_ids = gpu_ids
        # 初始化时，将 include_scale 参数传递给 DTCWTForward
        self.xfm = DTCWTForward(J=self.J, biort=biort, qshift=qshift, include_scale=self.include_scale).cuda(self.gpu_ids)

    def get_subband(self, subbands, layer, direction, real_imaginary):
        subband = subbands[layer][:, :, direction, :, :, real_imaginary]
        return subband

    # def forward(self, x):
    #     x = x.to(self.device)
    #     Yl, Yh = self.xfm(x)
    #     return Yl.float(), Yh.float()

    # # 将高频子带每一个方向的实部和虚部分别做MSE
    # def calculate1(self, subbands, layer):
    #     sum_Yh_real = 0
    #     sum_Yh_imag = 0
    #     for direction in range(6):
    #         Yh_real = self.get_subband(subbands, layer, direction, 0)
    #         Yh_imag = self.get_subband(subbands, layer, direction, 1)
    #         sum_Yh_real += Yh_real
    #         sum_Yh_imag += Yh_imag
    #     return sum_Yh_real, sum_Yh_imag
    #
    # def forward(self, x):
    #     x = x.to(self.device)
    #     Yl, Yh = self.xfm(x)
    #     # 将高频子带每一个方向的实部和虚部分别做MSE
    #     sum_Yh0_real, sum_Yh0_imag = self.calculate1(Yh, 0)
    #     sum_Yh1_real, sum_Yh1_imag = self.calculate1(Yh, 1)
    #     return Yl, sum_Yh0_real, sum_Yh0_imag, sum_Yh1_real, sum_Yh1_imag

    # # 将高频子带每一个方向的实部和虚部组合为复数做MSE
    # def calculate2(self, subbands, layer):
    #     sum_Yh_complex = 0
    #     for direction in range(6):
    #         Yh_real = self.get_subband(subbands, layer, direction, 0)
    #         Yh_imag = self.get_subband(subbands, layer, direction, 1)
    #         sum_Yh_complex += torch.complex(Yh_real, Yh_imag)
    #     return sum_Yh_complex
    #
    # def forward(self, x):
    #     x = x.to(self.device)
    #     Yl, Yh = self.xfm(x)
    #     # 将高频子带每一个方向的实部和虚部分别做MSE
    #     sum_Yh0_complex = self.calculate2(Yh, 0)
    #     sum_Yh1_complex = self.calculate2(Yh, 1)
    #     return Yl, sum_Yh0_complex, sum_Yh1_complex

    # # 直接用拼接的高频子带做MSE
    # def calculate3(self, subbands, layer):
    #
    #     return subbands[layer]
    #
    #
    # def forward(self, x):
    #     x = x.to(self.device)
    #     Yl, Yh = self.xfm(x)
    #     Yh0 = self.calculate3(Yh, 0)
    #     Yh1 = self.calculate3(Yh, 1)
    #     return Yl, Yh0, Yh1

    # 将高频子带每一个方向的实部和虚部分别做MSE
    def calculate4(self, subbands, layer):
        hf_reals = []       # 实部列表
        hf_imags = []       # 虚部列表
        for direction in range(6):
            hf_real = self.get_subband(subbands, layer, direction, 0)
            hf_imag = self.get_subband(subbands, layer, direction, 1)
            hf_reals.append(hf_real)
            hf_imags.append(hf_imag)
        return hf_reals, hf_imags

    def forward(self, x):
        x = x.cuda(self.gpu_ids)
        lf, hf = self.xfm(x)
        # 将高频子带每一个方向的实部和虚部分别做MSE
        hf0_reals, hf0_imags = self.calculate4(hf, 0)
        hf1_reals, hf1_imags = self.calculate4(hf, 1)
        return lf, hf0_reals, hf0_imags, hf1_reals, hf1_imags


# # 创建示例输入
# batch_size = 1
# channels = 3
# height = 4
# width = 4
# rec_A = torch.randn(batch_size, channels, height, width)
#
# if torch.cuda.is_available():
#     rec_A = rec_A.cuda(0)
#
# print(f"输入形状: {rec_A.shape}")
#
# """
# include_scale=False时，Yl 是最终（最小尺度）的 low，不是所有层的 low 列表。
# include_scale=True 时，函数会返回 scales（一个长度为 J 的 list）包含每一层的 low（这样你才能同时看到第 1 层 low 和第 2 层 low）。
# """
# wavelet_transform = DTCWTTransform2D(J=2, include_scale=False)
# Yl, Yh = wavelet_transform(rec_A)
#
# print(f"Yl 类型: {type(Yl)}")
# print(f"Yl 形状: {Yl.shape}")
# print(f"Yh 长度: {len(Yh)}")
# print(Yh[0])
# print(f"Yh[0] 形状: {Yh[0].shape}")
# print(f"Yh[1] 形状: {Yh[1].shape}")
# print(Yh[0][:, :, 0, :, :, 0])              # 第0层高频子带的，方向0的实部
# print(Yh[0][:, :, 0, :, :, 0])              # 第0层高频子带的，方向1的虚部
#
# real = Yh[0][:, :, 0, :, :, 0]
# imag = Yh[0][:, :, 0, :, :, 1]
# complex_one = torch.complex(real, imag)
# print(complex_one)
# print(f"Yh[0][:, :, 0, :, :, 0] : {Yh[0][:, :, 0, :, :, 0]}")
# print(f"Yh[0][:, :, 0, :, :, 1] : {Yh[0][:, :, 0, :, :, 1]}")
# print(f"complex_one: {complex_one}")

# 输入: (N, C, H, W)
# ↓ 第一级变换
# 低频: (N, C, H, W)
# 高频: (N, C, 6, H/2, W/2, 2)    # 6方向 × 实虚部
# ↓ 第二级变换
# 低频: (N, C, H/2, W/2)
# 高频: (N, C, 6, H/4, W/4, 2)
# ...

# DTCWT 在二维的一个重要性质是冗余是 2^d，二维则为 4:1（也就是说，输出系数的总数等于输入像素数的 4 倍）。
# H * W + 6 * H/2 * W/2 * 2(实虚部) = HW + 3HW，所以 DTCWT 是 4:1 冗余的。


# wavelet_transform = DTCWTTransform2D(J=2, include_scale=False)
# Yl, sum_Yh0_real, sum_Yh0_imag, sum_Yh1_real, sum_Yh1_imag = wavelet_transform(rec_A)
# print(f"Yl 形状: {Yl.shape}")
# print(f"sum_Yh0_real 形状: {sum_Yh0_real.shape}")
# print(f"sum_Yh0_imag 形状: {sum_Yh0_imag.shape}")
# print(f"sum_Yh1_real 形状: {sum_Yh1_real.shape}")
# print(f"sum_Yh1_imag 形状: {sum_Yh1_imag.shape}")

# wavelet_transform = DTCWTTransform2D(J=2, include_scale=False)
# Yl, sum_Yh0_complex, sum_Yh1_complex = wavelet_transform(rec_A)
# print(f"Yl 形状: {Yl.shape}")
# print(f"sum_Yh0_complex 形状: {sum_Yh0_complex.shape}")
# print(f"sum_Yh1_complex 形状: {sum_Yh1_complex.shape}")

# wavelet_transform = DTCWTTransform2D(J=2, include_scale=False)
# Yl, Yh0, Yh1 = wavelet_transform(rec_A)
# print(f"Yl 形状: {Yl.shape}")
# print(f"Yh0 形状: {Yh0.shape}")
# print(f"Yh1 形状: {Yh1.shape}")
