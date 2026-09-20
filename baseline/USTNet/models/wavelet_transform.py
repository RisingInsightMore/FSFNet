import torch
import torch.nn as nn
import numpy as np

"""
    标准 DWT 是无冗余的小波变换，系数数量与输入一致，方向性弱、平移不稳健；DTCWT 通过两棵（二维为四棵）滤波器树构造复小波以增强方向与平移不变性，因此产生 2^d的冗余（二维=4倍）。
    看看它与工具包输出的有什么区别，pywt（https://blog.csdn.net/wsp_1138886114/article/details/116780542）
    LL（低频），可视化：模糊的、缩小一半的图像；能看出大块的道路、楼群、农田和主体阴影方向。
             在 VIS→IR：代表整体热布局（城市区 vs 河流 vs 农田），对恢复地表温度分布、光照/阴影影响很有用。

    HL（水平细节），可视化：突出水平边缘（横向线条、横置屋檐、横向路标）。
             在 VIS→IR：对跑道、桥梁横向边缘、屋顶横向纹理敏感。
                
    LH（竖直细节），可视化：突出竖直边缘（建筑立面、灯杆、树干）。
             在 VIS→IR：对道路分割线、立柱型目标、建筑墙面边界敏感。

    HH（对角/点状高频），可视化：点状或角点、对角线纹理（屋顶瓦片、树冠、小车辆）。
             在 VIS→IR：对车辆轮廓、屋顶纹理、植被冠层细节等小尺度变化有响应。
             
    大尺度（LL）：常对应于地物的热平衡与大范围材料分布（例如：混凝土屋顶普遍比植被更暖/冷）。
                LL 能保留这些宏观对比与亮度趋势，是把 VIS 的整体布局（如建筑分布）翻译为 IR 的关键项。
    高频（LH/HL/HH）：负责把 VIS 中的几何边界（屋檐、道路边缘、树冠轮廓）映射为 IR 的边界与局部温差（例如，树冠边缘与地面温度差、车体与地面温差）。
                在实际航拍中很多重要目标（车辆、小建筑、水沟）都是由高频信息决定能否被正确生成/定位。
"""
class WaveletTransform2D(nn.Module):
    def __init__(self, in_channels, gpu_ids):
        super().__init__()
        self.in_channels = in_channels
        self.gpu_ids = gpu_ids

        # Haar wavelet filters，Haar小波的一维滤波系数，np.one((a,b))创建a行b列的矩阵
        self.harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))              # 低通滤波器（尺度函数）-求平均。[[1/√2, 1/√2]]
        self.harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))              # 高通滤波器（小波函数）-求差分。[[1/√2, 1/√2]]
        self.harr_wav_H[0, 0] = -1 * self.harr_wav_H[0, 0]              # 修改为[[-1/√2, 1/√2]]

        # 通过外积构造二维小波滤波器， np.transpose为转置
        """
             [[1/√2],  *   [[1/√2, 1/√2]] 
              [1/√2]]            
            =[[1/√2*1/√2, 1/√2*1/√2],
              [1/√2*1/√2, 1/√2*1/√2]]
            =[[0.5, 0.5],
              [0.5, 0.5]]
        """
        self.harr_wav_LL = np.transpose(self.harr_wav_L) * self.harr_wav_L          # LL：水平和垂直都低通，近视分量
        self.harr_wav_LH = np.transpose(self.harr_wav_L) * self.harr_wav_H          # LH：水平低通，垂直高通，水平细节
        self.harr_wav_HL = np.transpose(self.harr_wav_H) * self.harr_wav_L          # HL：水平高通，垂直低通，垂直细节
        self.harr_wav_HH = np.transpose(self.harr_wav_H) * self.harr_wav_H          # HH：水平和垂直都高通，对角细节

        # Convert filters to PyTorch tensors and move to device，unsqueeze(0)为在第0维添加一个维度，为了和输入的通道数一致
        """
            形状：      (2, 2)       -->    (1, 2, 2)
            内容：   [[0.5, 0.5],    -->    tensor([[[0.5, 0.5],
                     [0.5, 0.5]]                    [0.5, 0.5]]], device='cuda:0')
        """
        self.filter_LL = torch.from_numpy(self.harr_wav_LL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_LH = torch.from_numpy(self.harr_wav_LH).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HL = torch.from_numpy(self.harr_wav_HL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HH = torch.from_numpy(self.harr_wav_HH).unsqueeze(0).cuda(self.gpu_ids)

        # Define convolutional layers for each transform component，卷积核为2*2，步长为2，无填充，无偏置，分组卷积，每个通道独立处理 (保持通道独立性)
        # 定义下采样卷积，2x2卷积核对于Haar小波的2x2滤波器。即下采样(x-2)/2+1，尺寸变为x/2
        self.LL = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)
        self.LH = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)
        self.HL = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)
        self.HH = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)

        # Set weights to be non-trainable and assign filter coefficients，设置权重为不可训练，并分配滤波器系数
        with torch.no_grad():
            self.LL.weight.requires_grad = False        # 设置所有卷积的权重为不可训练
            self.LH.weight.requires_grad = False
            self.HL.weight.requires_grad = False
            self.HH.weight.requires_grad = False

            # 将预定义的Haar小波滤波器系数赋给卷积层的权重
            """
                self.filter_LL.float()           # 形状：(1, 2, 2)
                unsqueeze(0)                     # 形状：(1, 1, 2, 2)
                expand(in_channels, -1, -1, -1)  # 形状：(3, 1, 2, 2), in_channels表示将第一个维度扩展到3, -1表示保持原有维度不变
            """
            self.LL.weight.data = self.filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.LH.weight.data = self.filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.HL.weight.data = self.filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.HH.weight.data = self.filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            # print(self.LL.weight.data)
            # print(self.LL.weight.data.shape)



    def forward(self, input):
        LL = self.LL(input)
        LH = self.LH(input)
        HL = self.HL(input)
        HH = self.HH(input)
        # print(LL.shape)           # 尺寸都是变为了一半，因为做了一次卷积操作
        # print(LH.shape)

        return LL, LH, HL, HH


# # 创建示例输入
# batch_size = 1
# channels = 3
# height = 256
# width = 256
#
# # 创建随机输入张量 (模拟rec_A)
# rec_A = torch.randn(batch_size, channels, height, width)
#
# # 如果有GPU，将张量移到GPU上
# if torch.cuda.is_available():
#     rec_A = rec_A.cuda(0)
#
# print(f"Input shape: {rec_A.shape}")
#
# # 创建小波变换实例并进行测试
# wavelet_transform = WaveletTransform2D(3, 0)
# LL, LH, HL, HH = wavelet_transform(rec_A)



