import numpy as np
import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models.networks import *
from collections import OrderedDict
import util.util as util
from models.losses import *


class USTNet(BaseModel):
    def name(self):
        return 'USTNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.use_pretrain = opt.use_pretrain

        # self.input_A = self.Tensor(nb, opt.input_nc, size, size)            # 可见光
        # self.input_B = self.Tensor(nb, opt.output_nc, size, size)           # 红外图像
        self.input_A = torch.zeros((nb, opt.input_nc, size, size)).cuda(opt.gpu_ids[0])           # 可见光
        self.input_B = torch.zeros((nb, opt.output_nc, size, size)).cuda(opt.gpu_ids[0])          # 红外图像

        # generator
        self.netG_AB = USTNetModel(opt).cuda(opt.gpu_ids[0])                # 可见光到红外的生成器
        self.netD_B = NLayerDiscriminator(opt).cuda(opt.gpu_ids[0])         # 红外的判别器
        self.netD_gc_B = NLayerDiscriminator(opt).cuda(opt.gpu_ids[0])      # 红外的全局判别器（旋转后）

        self.init_weights()

        if not self.isTrain:                                                # 测试的话，直接加载模型
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch, opt.gpu_ids)
            self.load_network(self.netD_B, 'D_B', which_epoch, opt.gpu_ids)
            self.load_network(self.netD_gc_B, 'D_gc_B', which_epoch, opt.gpu_ids)
        if self.use_pretrain:                                              # 预训练，直接加载G_AB生成器
            which_epoch = opt.which_epoch
            print('load model from:{}'.format(self.save_pretrain_dir))
            self.load_pretrain_network(self.netG_AB, 'G_AB', which_epoch, opt.gpu_ids)      # 在随机掩码上训练的USTNetModel（包含参数，后续操作则是基于此微调）

        if self.isTrain:
            self.criterionGAN = GANLoss(gan_mode=opt.gan_mode).cuda(opt.gpu_ids[0])     # 对抗损失，用于生成器/判别器的真假辨别,lsgan最小二乘GAN使用MSE损失
            self.criterionIdt = torch.nn.L1Loss()                                       # 身份损失，用于保持输入和输出的相似性（当identity＞0时，若输入真实B，输出也应接近真实B）
            self.criterionGc = torch.nn.L1Loss()                                        # 几何一致性损失，用于约束几何变换前后结果的一致性

            self.fake_B_pool = ImagePool(opt.pool_size)                 # 构建fake_B的图像池，缓存假图像以稳定鉴别器训练
            self.fake_gc_B_pool = ImagePool(opt.pool_size)              # 构建fake_gc_B的图像池

            self.model_names = ['G_AB', 'D_B', 'D_gc_B']

            # initialize optimizers，优化器
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(
                                itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))

            # initialize schedulers，线性学习率调度器，先预热，再衰减
            self.scheduler_G = get_scheduler(self.optimizer_G, opt)
            self.scheduler_D_B = get_scheduler(self.optimizer_D_B, opt)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']           # 可见光
        input_B = input['B' if AtoB else 'A']           # 红外图像
        self.input_A.resize_(input_A.size()).copy_(input_A)         # 将内部输入形状调整为外部输入的形状，[nb, nc, size, size]
        self.input_B.resize_(input_B.size()).copy_(input_B)         # 将外部输入复制到内部预分配的 Tensor
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # 存储输入可见光路径，用于后续可视化

    """复制输入，并根据rot、vf、hf进行几何变换"""
    def forward(self):
        input_A = self.input_A.clone()                  # 复制输入，用于后续的旋转变换
        input_B = self.input_B.clone()                  # 形状为[nb, nc, size, size]

        self.real_A = self.input_A                      # 真实的可见光图像，形状为[nb, nc, size, size]
        self.real_B = self.input_B                      # 真实的红外图像，形状为[nb, nc, size, size]

        size = self.opt.fineSize                        # 裁剪后图像的大小

        if self.opt.geometry == 'rot':                  # 旋转90度（转置+反向索引）
            self.real_gc_A = self.rot90(input_A, 0)     # 先转置, 第二维度和第三维度互换 （[nb, nc, size, size] --> [nb, nc, size, size]）
            self.real_gc_B = self.rot90(input_B, 0)     # 后水平翻转 （[nb, nc, size, size] --> [nb, nc, size, size]）
        elif self.opt.geometry == 'vf':                 # 竖直翻转（沿第二维度反转）
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])
            self.real_gc_A = torch.index_select(input_A, 2, inv_idx)    # 在第2维上按反转索引重新排列，张量进行垂直翻转
            self.real_gc_B = torch.index_select(input_B, 2, inv_idx)    # （[nb, nc, size, size] --> [nb, nc, size, size]）
        elif self.opt.geometry == 'hf':                 # 水平翻转（沿第三维度反转）
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])
            self.real_gc_A = torch.index_select(input_A, 3, inv_idx)    # 在第3维上按反转索引重新排列，张量进行水平翻转
            self.real_gc_B = torch.index_select(input_B, 3, inv_idx)    # （[nb, nc, size, size] --> [nb, nc, size, size]）
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

    def backward_G(self):
        # print()
        # 原始图像
        fake_B = self.netG_AB.forward(self.real_A)      # 输入生成器得到生成图像，[nb, input_nc, size, size]  -> [nb, output_nc, size, size]
        pred_fake = self.netD_B.forward(fake_B)         # 输入判别器得到判别结果，[nb, output_nc, size, size] -> [nb, 1, size/8, size/8]
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        # 旋转后的图像
        fake_gc_B = self.netG_AB.forward(self.real_gc_A)    # [nb, input_nc, size, size]  -> [nb, output_nc, size, size]
        pred_fake = self.netD_gc_B.forward(fake_gc_B)       # [nb, output_nc, size, size] -> [nb, 1, size/8, size/8]
        loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        if self.opt.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)
        else:
            loss_gc = self.get_gc_hf_loss(fake_B, fake_gc_B)

        if self.opt.identity > 0:
            # G_AB should be identity if real_B is fed.
            idt_A = self.netG_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            idt_gc_A = self.netG_AB(self.real_gc_B)
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

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc

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
            # print(True)
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

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)             # 转置，第2维度和第3维度互换（[nb, bc, h, w] --> [nb, bc, w, h]）
        size = self.opt.fineSize                    # 剪裁尺寸大小
        inv_idx = torch.arange(size-1, -1, -1).long().cuda(self.opt.gpu_ids[0])     # 创建一个递减序列（size-1，size-2，... 0）
        if direction == 0:
            tensor = torch.index_select(tensor, 3, inv_idx)         # 在第3维上按反转索引重新排列，转置后的张量进行水平翻转（转置+翻转即相当于顺时针90度旋转）
        else:
            tensor = torch.index_select(tensor, 2, inv_idx)         # 在第2维上按反转索引重新排列，转置后的张量进行垂直翻转（转置+翻转即相当于顺时针90度旋转）
        return tensor            # [nb, bc, w, h]

    def get_image_paths(self):
        return self.image_paths

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
            AB_gt = self.rot90(AB_gc.clone().detach(), 1)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 0)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
            AB_gt = self.rot90(AB_gc.clone().detach(), 0)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 1)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc

        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        return loss_gc

    def get_gc_hf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])

        AB_gt = torch.index_select(AB_gc.clone().detach(), 3, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 3, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc

        return loss_gc

    def get_current_errors(self):

        ret_errors = OrderedDict(
            [('D_B', self.loss_D_B), ('D_gc_B', self.loss_D_gc_B),
             ('G_AB', self.loss_G_AB), ('G_gc_AB', self.loss_G_gc_AB),
             ('Gc', self.loss_gc)])

        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc

        return ret_errors

    def set_requires_grad(self, models, requires_grad=False):
        # pylint: disable=no-self-use
        if not isinstance(models, list):
            models = [models, ]

        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()
        # update G
        self.set_requires_grad([self.netD_B, self.netD_gc_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update D

        self.set_requires_grad([self.netD_B, self.netD_gc_B], True)
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.opt.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.opt.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_gc_B', label, self.opt.gpu_ids)

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
        self.fake_B = self.netG_AB(self.real_A)

    def update_learning_rate(self):
        self.scheduler_G.step()
        self.scheduler_D_B.step()
        lr = self.optimizer_D_B.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
