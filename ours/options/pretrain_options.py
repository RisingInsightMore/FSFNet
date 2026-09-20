# --*--coding   : uft-8  --*--
# @Time         : 2026/2/16 10:50
# @Author       : RisingInsight
# @File         : pretrain_options2.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import argparse
import os
import time
import random
import numpy as np
import torch
from util import util
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
        # 原有参数（保持不变）
        self.parser.add_argument('--dataroot', default='../Datasets/DayDrone',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=288, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
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
        self.parser.add_argument('--model', type=str, default='FSFNetPT',
                                 help='chooses which model to use.')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize|resize_and_crop|crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--lr_policy', type=str, default='CosineAnnealingWarmRestarts',
                                 help='learning rate policy: linear')
        self.parser.add_argument('--patch_size', type=int, default=16,
                                 help='patch size for masking')
        self.parser.add_argument('--fraction', type=float, default=0.4,
                                 help='patch size for masking')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training from the latest checkpoint')
        self.parser.add_argument('--epochs', type=int, default=400,
                                 help='epochs for training')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, '
                                      '<epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory '
                                      'contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--load_dir', type=str, default='',
                                 help='path to load checkpoints from (for resuming training). If empty, use save_dir.')
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
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--use_wavelet', action='store_true',
                                 help='If specified, use wavelet loss in pretrain')
        self.parser.add_argument('--pretrain_name', type=str, default='experiment',
                                 help='name of the pretrain. It decides where to load pretrain model')
        # self.parser.add_argument('--use_pretrain', type=bool, default=True,
        #                          help='If using pretrain, set True')
        # self.parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='weight for perceptual loss')
        self.parser.add_argument('--lambda_wavelet', type=float, default=1.0, help='weight for wavelet loss')
        self.parser.add_argument('--lambda_aux', type=float, default=0.1, help='weight for auxiliary loss')
        # 新增参数：name_id 和 experiment_id，用户可指定数字，不指定则自动递增
        self.parser.add_argument('--name_id', type=int, default=None,
                                 help='ID number for the model variant under the same --name. Auto-incremented if not provided.')
        self.parser.add_argument('--experiment_id', type=int, default=None,
                                 help='ID number for the experiment under the same name_id. Auto-incremented if not provided.')

        self.isTrain = True
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        # 处理 GPU IDs
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # ==================== 自动编号逻辑 ====================
        base_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(base_dir, exist_ok=True)

        # 1. 确定 name_id
        if self.opt.name_id is None:
            # 自动递增：找到所有以 self.opt.name 开头且后面全是数字的目录
            existing = []
            for d in os.listdir(base_dir):
                full_path = os.path.join(base_dir, d)
                if os.path.isdir(full_path) and d.startswith(self.opt.name):
                    suffix = d[len(self.opt.name):]
                    if suffix.isdigit():
                        existing.append(int(suffix))
            max_id = max(existing) if existing else -1
            name_id_val = max_id + 1
        else:
            name_id_val = self.opt.name_id

        name_id_str = f"{self.opt.name}{name_id_val}"           # 例如 AVIID_FSFNet_Pretrain0
        name_dir = os.path.join(base_dir, name_id_str)
        os.makedirs(name_dir, exist_ok=True)                    # 例如：./checkpoints/FSFNet_Pretrain/FSFNet_Pretrain0

        # # 2. 确定 experiment_id
        # if self.opt.experiment_id is None:
        #     # 自动递增：在 name_dir 下找所有以 name_id_str + "_experiment" 开头且后面是数字的目录
        #     prefix = f"{name_id_str}_experiment"
        #     existing_exp = []
        #     for d in os.listdir(name_dir):
        #         full_path = os.path.join(name_dir, d)
        #         if os.path.isdir(full_path) and d.startswith(prefix):
        #             suffix = d[len(prefix):]
        #             if suffix.isdigit():
        #                 existing_exp.append(int(suffix))
        #     max_exp_id = max(existing_exp) if existing_exp else -1
        #     experiment_id_val = max_exp_id + 1
        # else:
        #     experiment_id_val = self.opt.experiment_id

        # experiment_dir_name = f"{name_id_str}_experiment{experiment_id_val}"    # 例如 AVIID_FSFNet_Pretrain0_experiment0
        # experiment_dir = os.path.join(name_dir, experiment_dir_name)            # 例如 ./checkpoints/FSFNet_Pretrain/FSFNet_Pretrain0/FSFNet_Pretrain0_experiment0
        # os.makedirs(experiment_dir, exist_ok=True)

        # 将生成的ID值和目录信息保存到 opt 中，方便其他地方使用
        # self.opt.name_id = name_id_val
        # self.opt.experiment_id = experiment_id_val
        # self.opt.name_id_dir = name_id_str
        self.opt.name_dir = name_dir
        # self.opt.experiment_dir = experiment_dir
        # ==================== 结束 ====================

        # 打印所有选项
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # 保存选项到文件
        opt_file_path = os.path.join(self.opt.name_dir, 'opt.txt')
        with open(opt_file_path, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt





