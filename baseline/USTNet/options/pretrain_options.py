import argparse
import os
from util import util
import torch
import time
import random
import numpy as np
from config import config


class Options():
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
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')                   # 16
        self.parser.add_argument('--loadSize', type=int, default=288, help='scale images to this size')         # 数据加载线程288
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')            # 图像分辨率256
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0,2 use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned',
                                 help='chooses how datasets are loaded. [unaligned | aligned | single]')
        # self.parser.add_argument('--model', type=str, default='USTNetPT',
        #                          help='chooses which model to use.')
        self.parser.add_argument('--model', type=str, default='USTNetPT2',
                                 help='chooses which model to use.')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')          # 8
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--resize_or_crop', type=str, default='pretrain',
                                 help='scaling and cropping of images at load time [resize|resize_and_crop|crop]')      # 图像预处理方式，缩放，裁剪，颜色抖动
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--lr_policy', type=str, default='CosineAnnealingWarmRestarts',
                                 help='learning rate policy: linear ')                  # 学习率策略改为余弦退火热重启（更先进的调度策略）
        self.parser.add_argument('--patch_size', type=int, default=16,
                                 help='patch size for masking ')                        # 掩码的patch大小（MAE的预训练任务）
        self.parser.add_argument('--fraction', type=float, default=0.4,
                                 help='patch size for masking ')                        # 掩码比例（40%的patch会被掩码）
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epochs', type=int, default=400,
                                 help='epochs for training ')                           # 总训练epoch数增加到400
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, '
                                      '<epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory '
                                      'contains more than max_dataset_size, only a subset is loaded.')

        self.parser.add_argument('--display_freq', type=int, default=10,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain '
                                      'number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=200,
                                 help='frequency of saving checkpoints at the end of epochs')           # 保存频率调整为每200个epoch（之前是100）
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        self.parser.add_argument('--use_wavelet', type=bool, default=True,
                                 help='If using wavelet for pretrain, set True')                        # 是否使用小波变换进行预训练
        self.parser.add_argument('--pretrain_name', type=str, default='experiment',
                                 help='name of the pretrain. It decides where to load pretrain model')
        self.parser.add_argument('--use_pretrain', type=bool, default=True,
                                 help='If using pretrain, set True')                                    # 是否使用预训练（这里设为True，表示这是预训练阶段）
        self.parser.add_argument('--lambda_wavelet', type=float, default=1.0, help='weight for wavelet loss')       # 小波损失的权重

        self.isTrain = True
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
        file_name = os.path.join(expr_dir, 'pretrain_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
