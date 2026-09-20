import random
import numpy as np
import torch
from torch.autograd import Variable

"""
    图像池类，用于存储和管理生成的图像
"""
class ImagePool():
    def __init__(self, pool_size):              # 初始化图像池
        self.pool_size = pool_size              # 图像池大小
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):                    # 查询图像池
        if self.pool_size == 0:
            return Variable(images)             # 图像池大小为0，直接返回输入图像
        return_images = []

        # 遍历输入图像，将其添加到图像池或从图像池中随机选择图像
        for image in images:
            image = torch.unsqueeze(image, 0)   # 3D -> 4D: [C,H,W] -> [1,C,H,W]
            if self.num_imgs < self.pool_size:      # 图像池未满，直接添加图像
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)       # 随机数，用于判断是否从图像池中选择图像
                if p > 0.5:                         # 50%概率使用池中图像，从图像池中随机选择一个图像
                    random_id = random.randint(0, self.pool_size-1)     # 随机选择一个图像池中的图像
                    tmp = self.images[random_id].clone()     # 保存池中旧图像
                    self.images[random_id] = image           # 用新图像替换
                    return_images.append(tmp)                # 返回旧图像
                else:
                    return_images.append(image)     # 50%概率直接使用新图像
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
