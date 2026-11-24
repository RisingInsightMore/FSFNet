from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # 4D张量（batch, channels, height, width）
    # 这里使用[0]取第一个样本，所以变成了3D张量（channels, height, width），然后转化为Numpy数组
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:       # 如果是单通道图像，复制为3通道（灰度转RGB），伪彩色图
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # 将通道顺序从 (channels, height, width) 转换为 (height, width, channels)
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) + 1      # +1 从[-1, 1]归一化到[0, 2]
    image_numpy = image_numpy / 2.0         # 从[0, 2]归一化到[0, 1]
    image_numpy = image_numpy * 255.0       # 从[0, 1]归一化到[0, 255]
    return image_numpy.astype(imtype)       # 转换为指定数据类型（默认uint8）


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():          # 遍历网络所有参数
        if param.grad is not None:          # 只统计有梯度的参数
            mean += torch.mean(torch.abs(param.grad.data))       # 计算参数梯度的绝对值的均值
            count += 1
    if count > 0:               # 如果有参数有梯度，计算平均梯度
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)    # 从Numpy数组创建PIL图像对象
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    # 遍历调用该函数的那一行代码所在上下文的所有代码行，使用正则表达式搜索匹配 varname(变量名) 的调用模式
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)           # 返回第一个捕获组的内容，即变量名部分


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
