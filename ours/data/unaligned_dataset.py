import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

# 非配对数据集
class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot + '/', opt.phase + 'A')        # 构建域A和域B的数据目录路径
        self.dir_B = os.path.join(opt.dataroot + '/', opt.phase + 'B')

        print(f"dir_A: {self.dir_A}")

        self.A_paths = make_dataset(self.dir_A)                         # 获取域A和域B中所有图像文件的路径列表
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)                             # 对图像路径进行排序，确保每次加载顺序一致
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)                                 # 记录域A和域B的图像数量
        self.B_size = len(self.B_paths)

        is_test = (opt.phase == 'test')
        self.transform = get_transform(opt, is_test=is_test)                             # 获取图像预处理变换（如缩放、裁剪、归一化等）

    # 实现数据集索引访问，获取单个数据样本
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]                  # 获取域A中对应索引的图像路径（使用取模运算确保索引不越界）
        index_A = index % self.A_size                               # 记录实际的域A索引

        # 确定域B的索引：如果使用顺序批次，域B索引与域A索引对应；否则随机选择域B的一个索引（实现非配对数据）
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]                          # 获取域B中对应索引的图像路径
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')               # 打开域A和域B的图像文件，并转换为RGB格式
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)                               # 对图像应用预处理变换（转为Tensor、归一化等）
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':                  # 如果是从B到A的方向，输入是B（输出通道数），输出是A（输入通道数）
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # 如果输入需要单通道（灰度图），将RGB三通道图像转换为单通道灰度图
        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114     # Y = 0.299R + 0.587G + 0.114B
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
