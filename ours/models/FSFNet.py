# ----coding   : utf-8  ----
# @Time         : 2026/6/19 9:24
# @Author       : RisingInsight
# @File         : FSFNet4.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import numpy as np
import torch
import os
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models.networks import *
from collections import OrderedDict
import util.util as util
from models.losses import *
import torch.nn.functional as F

class FSFNet(BaseModel):
    def name(self):
        return 'FSFNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.use_pretrain = getattr(opt, 'use_pretrain', False)

        self.input_A = torch.zeros((nb, opt.input_nc, size, size)).cuda(opt.gpu_ids[0])
        self.input_B = torch.zeros((nb, opt.output_nc, size, size)).cuda(opt.gpu_ids[0])

        # 初始化生成器和判别器
        self.netG_AB = FSFNetModel(opt).cuda(opt.gpu_ids[0])
        self.netD_B = NLayerDiscriminator(opt).cuda(opt.gpu_ids[0])
        self.netD_gc_B = NLayerDiscriminator(opt).cuda(opt.gpu_ids[0])

        self.init_weights()

        # 频率域相关（用于频率风格损失）
        self.wavelet_transform = MultiLevelWaveletTransform(
            opt.input_nc, opt.gpu_ids[0], levels=2
        )

        # 设置模型名称列表（无论训练还是测试都需要）
        self.model_names = ['G_AB', 'D_B', 'D_gc_B']

        if self.isTrain:
            # 训练模式：设置起始 epoch
            self.epoch = getattr(opt, 'epoch_count', 1)

            self.criterionGAN = GANLoss(gan_mode=opt.gan_mode).cuda(opt.gpu_ids[0])
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            self.criterionFreq = torch.nn.MSELoss()  # 用于频率风格损失
            self.criterionRes = torch.nn.L1Loss()    # 用于辅助重建损失（频率路径）

            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_gc_B_pool = ImagePool(opt.pool_size)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(
                                itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))

            self.scheduler_G = get_scheduler(self.optimizer_G, opt)
            self.scheduler_D_B = get_scheduler(self.optimizer_D_B, opt)

            if opt.continue_train:
                self.load_checkpoint(opt.which_epoch)

        if not self.isTrain:
            # 测试模式：加载指定 epoch 的权重
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch, opt.gpu_ids)
            self.load_network(self.netD_B, 'D_B', which_epoch, opt.gpu_ids)
            self.load_network(self.netD_gc_B, 'D_gc_B', which_epoch, opt.gpu_ids)

        if self.use_pretrain:
            if hasattr(opt, 'pretrain_dir') and opt.pretrain_dir:
                self.load_pretrain_dir = opt.pretrain_dir
            else:
                raise ValueError("pretrain_dir must be specified when use_pretrain=True")
            which_epoch = opt.which_epoch
            self.load_pretrain_network(self.netG_AB, 'G_AB', which_epoch, opt.gpu_ids)

    # ---------- 断点续训相关方法 ----------
    def save_checkpoint(self, epoch_label):
        save_filename = 'latest_checkpoint.pth' if epoch_label == 'latest' else f'{epoch_label}_checkpoint.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        checkpoint = {
            'epoch': self.epoch,
            'model_G_AB_state_dict': self.netG_AB.state_dict(),
            'model_D_B_state_dict': self.netD_B.state_dict(),
            'model_D_gc_B_state_dict': self.netD_gc_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_B_state_dict': self.scheduler_D_B.state_dict(),
            'opt': self.opt
        }
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved to {save_path}')

    def load_checkpoint(self, epoch_label):
        if epoch_label == 'latest':
            checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'{epoch_label}_checkpoint.pth')

        if os.path.isfile(checkpoint_path):
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage.cuda(self.opt.gpu_ids[0]),
                                    weights_only=False)
            self.netG_AB.load_state_dict(checkpoint['model_G_AB_state_dict'])
            self.netD_B.load_state_dict(checkpoint['model_D_B_state_dict'])
            self.netD_gc_B.load_state_dict(checkpoint['model_D_gc_B_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D_B.load_state_dict(checkpoint['scheduler_D_B_state_dict'])
            self.epoch = checkpoint['epoch']
            print(f'Resumed from epoch {self.epoch}')
        else:
            print(f'No checkpoint found at {checkpoint_path}, starting from scratch.')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # ---------- 修正的几何变换函数 ----------
    def rot90(self, tensor, k=1):
        """逆时针旋转90度k次"""
        return torch.rot90(tensor, k, dims=[2, 3])

    def vf(self, tensor):
        """垂直翻转"""
        return torch.flip(tensor, dims=[2])

    def hf(self, tensor):
        """水平翻转"""
        return torch.flip(tensor, dims=[3])

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()
        self.real_A = self.input_A
        self.real_B = self.input_B
        size = self.opt.fineSize

        # 应用几何变换
        if self.opt.geometry == 'rot':
            self.real_gc_A = self.rot90(input_A, 1)  # 逆时针90度
            self.real_gc_B = self.rot90(input_B, 1)
        elif self.opt.geometry == 'vf':
            self.real_gc_A = self.vf(input_A)
            self.real_gc_B = self.vf(input_B)
        elif self.opt.geometry == 'hf':
            self.real_gc_A = self.hf(input_A)
            self.real_gc_B = self.hf(input_B)
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

    def backward_G(self):
        # Compute granular alpha for ramp-in
        if getattr(self.opt, 'use_granular', False) and hasattr(self.netG_AB, 'set_granular_alpha'):
            if self.epoch < self.opt.epochs_warmup:
                alpha = 0.0
            else:
                ramp_epochs = self.opt.epochs_warmup // 2
                alpha = min(1.0, (self.epoch - self.opt.epochs_warmup) / ramp_epochs)
            self.netG_AB.set_granular_alpha(alpha)
            self.granular_alpha = alpha

        # 接收生成器的两个输出 (spatial_out, freq_out)
        fake_B, fake_freq_B = self.netG_AB(self.real_A)

        # 主输出对抗损失
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        # 几何变换分支（仅用主输出）
        if self.opt.geometry == 'rot':
            fake_gc_B, _ = self.netG_AB(self.real_gc_A)
            pred_fake = self.netD_gc_B.forward(fake_gc_B)
            loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 1)
        elif self.opt.geometry == 'vf':
            fake_gc_B, _ = self.netG_AB(self.real_gc_A)
            pred_fake = self.netD_gc_B.forward(fake_gc_B)
            loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)
        else:  # hf
            fake_gc_B, _ = self.netG_AB(self.real_gc_A)
            pred_fake = self.netD_gc_B.forward(fake_gc_B)
            loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G
            loss_gc = self.get_gc_hf_loss(fake_B, fake_gc_B)

        # Identity Loss（仅用主输出）
        if self.opt.identity > 0:
            idt_A, _ = self.netG_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            idt_gc_A, _ = self.netG_AB(self.real_gc_B)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.lambda_AB * self.opt.identity
            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        # 新增：辅助频率重建损失（迫使Transformer学习有效频率特征）
        loss_aux = self.criterionRes(fake_freq_B, self.real_B) * self.opt.lambda_aux
        self.loss_aux = loss_aux.item()

        # 频率风格损失（小波统计量匹配，默认启用）
        if getattr(self.opt, 'use_wavelet_loss', True):
            fake_wavelets = self.wavelet_transform(fake_B)   # 注意这里仍用主输出计算小波损失
            real_wavelets = self.wavelet_transform(self.real_B)
            loss_freq = 0
            for lvl, (fake_subs, real_subs) in enumerate(zip(fake_wavelets, real_wavelets)):
                for fake_sub, real_sub in zip(fake_subs, real_subs):
                    fake_mean = fake_sub.mean([2,3], keepdim=True)
                    fake_std = fake_sub.std([2,3], keepdim=True)
                    real_mean = real_sub.mean([2,3], keepdim=True)
                    real_std = real_sub.std([2,3], keepdim=True)
                    loss_freq += self.criterionFreq(fake_mean, real_mean) + self.criterionFreq(fake_std, real_std)
            loss_freq = loss_freq * getattr(self.opt, 'lambda_freq', 0.1)
            self.loss_freq = loss_freq.item()
        else:
            loss_freq = 0
            self.loss_freq = 0

        # 总损失
        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc + loss_freq + loss_aux
        loss_G.backward()

        self.fake_B = fake_B
        self.fake_gc_B = fake_gc_B
        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB = loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_real = self.criterionGAN(pred_real, True)

        with torch.no_grad():
            fake = fake.contiguous()
        pred_fake = netD(fake)
        loss_fake = self.criterionGAN(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5

        if self.opt.gradient_penalty:
            loss += cal_gradient_penalty(
                netD, real, fake, real.device, constant=self.opt.constant, lambda_gp=self.opt.lambda_gp
            )[0]

        loss.backward()
        return loss

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)
        loss_D_gc_B = self.backward_D_basic(self.netD_gc_B, self.real_gc_B, fake_gc_B)
        self.loss_D_B = loss_D_B.item()
        self.loss_D_gc_B = loss_D_gc_B.item()

    # ---------- 修正的几何一致性损失计算 ----------
    def get_gc_rot_loss(self, AB, AB_gc, k):
        """k为旋转次数，需与forward一致"""
        loss_gc = 0.0
        AB_gt = self.rot90(AB_gc.clone().detach(), -k)
        loss_gc = self.criterionGc(AB, AB_gt)
        AB_gc_gt = self.rot90(AB.clone().detach(), k)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        AB_gt = self.vf(AB_gc.clone().detach())
        loss_gc = self.criterionGc(AB, AB_gt)
        AB_gc_gt = self.vf(AB.clone().detach())
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        return loss_gc

    def get_gc_hf_loss(self, AB, AB_gc):
        AB_gt = self.hf(AB_gc.clone().detach())
        loss_gc = self.criterionGc(AB, AB_gt)
        AB_gc_gt = self.hf(AB.clone().detach())
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        return loss_gc

    def get_image_paths(self):
        return self.image_paths

    def get_current_errors(self):
        ret_errors = OrderedDict(
            [('D_B', self.loss_D_B), ('D_gc_B', self.loss_D_gc_B),
             ('G_AB', self.loss_G_AB), ('G_gc_AB', self.loss_G_gc_AB),
             ('Gc', self.loss_gc), ('Aux', self.loss_aux)])
        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc
        if getattr(self.opt, 'use_wavelet_loss', False):
            ret_errors['freq'] = self.loss_freq
        if getattr(self.opt, 'use_granular', False):
            ret_errors['gr_alpha'] = getattr(self, 'granular_alpha', 0.0)
        return ret_errors

    def set_requires_grad(self, models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()
        # 更新 G
        self.set_requires_grad([self.netD_B, self.netD_gc_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 更新 D
        self.set_requires_grad([self.netD_B, self.netD_gc_B], True)
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.opt.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.opt.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_gc_B', label, self.opt.gpu_ids)
        self.save_checkpoint(label)   # 新增：保存优化器状态

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.detach().clone())
        real_B = util.tensor2im(self.real_B.detach().clone())
        fake_B = util.tensor2im(self.fake_B.detach().clone())
        ret_visuals = OrderedDict(
            [('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        return ret_visuals

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        outputs = self.netG_AB(self.real_A)
        if isinstance(outputs, tuple):
            self.fake_B = outputs[0]  # 取主图像
        else:
            self.fake_B = outputs

    def update_learning_rate(self):
        self.scheduler_G.step()
        self.scheduler_D_B.step()
        lr = self.optimizer_D_B.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def _init_weights(self, m):
        if hasattr(m, 'weight') and m.weight is not None:
            if not m.weight.requires_grad:
                return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def init_weights(self):
        self.netG_AB.apply(self._init_weights)
        self.netD_B.apply(self._init_weights)
        self.netD_gc_B.apply(self._init_weights)