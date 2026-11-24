import argparse
import os
from util import util
import time
import torch
import random
import numpy as np
from config import config

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.set_seed(42)

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def initialize(self):
        self.parser.add_argument('--dataroot', default='../Datasets/DayDrone',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')                   # 8，批处理大小，每次训练处理的图像数量
        self.parser.add_argument('--loadSize', type=int, default=288, help='scale images to this size')         # 288，图像加载时的缩放尺寸
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')            # 256，随机裁剪后的最终输入尺寸（数据增强）
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')           # 输入图像通道数（RGB可见光图像）
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')         # 输出图像通道数（红外图像，虽然红外通常是单通道，这里做了三通道处理）
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')    # 生成器第一层卷积的滤波器数量
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')        # 判别器第一层卷积的滤波器数量
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')   # 判别器的卷积层数
        self.parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0,2. use -1 for CPU')          #
        self.parser.add_argument('--name', type=str, default='experiment',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned',
                                 help='chooses how datasets are loaded. [unaligned | aligned | single]')                    # 非配对数据集
        self.parser.add_argument('--model', type=str, default='USTNet',
                                 help='chooses which model to use.')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')            # 转换方向：A域到B域（可见光→红外）
        self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')          # 16，数据加载线程
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')          # 归一化方式：实例归一化（适合风格迁移任务）
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')          # 是否按顺序而非随机取batch（是否可见光和红外图像配对）。如果命令行中指定了这个参数，其值即为 True，否则为 False
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')           # 显示窗口大小
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')         # Visdom显示窗口ID
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')   # Visdom服务端口
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')        # 生成器是否使用dropout，默认未使用
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')      # 最大数据集大小（用于调试时限制数据量）
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize|resize_and_crop|crop]')             # 图像预处理方式：先缩放再随机裁剪
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')                            # 是否禁用水平翻转（数据增强）
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')    # 是否不保存训练过程中的HTML可视化结果
        # self.parser.add_argument('--init_type', type=str, default='xavier',
        #                          help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--pretrain_name', type=str, default='experiment',
                                 help='name of the pretrain. It decides where to load pretrain model')                         # 预训练模型名称
        self.parser.add_argument('--use_pretrain', type=bool, default=False,
                                 help='If using pretrain, set True')                                                           # 是否使用预训练模型

        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # 为每个实验创建唯一标识
        if hasattr(self.opt, 'experiment_id'):
            self.experiment_id = self.opt.experiment_id
        else:
            self.experiment_id = f"{self.opt.name}_{config.timestamp}"
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.experiment_id)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
