###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# 判断文件是否为图像文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 创建数据集函数：递归遍历目录，收集所有图像文件路径
def make_dataset(dir):
    images = []
    # 确保输入路径是一个有效的目录
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # 递归遍历目录及其所有子目录，os.walk返回三元组：(当前目录路径, 子目录列表, 文件列表)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:                        # 遍历当前目录下的所有文件
            if is_image_file(fname):                # 检查文件是否为图像文件
                path = os.path.join(root, fname)    # 构建完整的文件路径
                images.append(path)                 # 将路径添加到图像列表中

    return images


# 默认图像加载器函数
def default_loader(path):
    return Image.open(path).convert('RGB')          # 打开图像文件并转换为RGB格式（确保3通道）



class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)                # 获取目录中所有图像文件的路径
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    # 实现索引访问，获取单个数据样本
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
