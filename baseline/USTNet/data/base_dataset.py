import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

"""
尺寸调整和裁剪, 数据增强
"""
def get_transform(opt, is_test=False):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':                         # 缩放并随机裁剪
        osize = opt.loadSize
        transform_list.append(transforms.Resize((osize, osize)))
        if is_test:
            transform_list.append(transforms.CenterCrop(opt.fineSize))  # 测试时使用中心裁剪，裁剪尺寸为 fineSize
        else:
            transform_list.append(transforms.RandomCrop(opt.fineSize))  # 训练时随机裁剪，裁剪尺寸为 fineSize
    elif opt.resize_or_crop == 'resize':                               # 缩放
        osize = opt.loadSize
        transform_list.append(transforms.Resize((osize, osize)))
    elif opt.resize_or_crop == 'crop':                                # 随机裁剪
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'pretrain':                            # 预训练时的数据增强
        osize = opt.loadSize
        transform_list.append(transforms.Resize((osize, osize)))      # 缩放
        transform_list.append(transforms.RandomRotation(degrees=10))  # 随机旋转
        fsize = opt.fineSize
        transform_list.append(transforms.RandomCrop(fsize))           # 随机裁剪
        # 颜色抖动（亮度、对比度、饱和度和色调的随机调整）
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())      # 随机水平翻转

    # if opt.isTrain and opt.rotation:

    transform_list += [transforms.ToTensor(),                        # 转换为张量
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]    # 归一化[-1,1]
    return transforms.Compose(transform_list)                        # 组合变换
