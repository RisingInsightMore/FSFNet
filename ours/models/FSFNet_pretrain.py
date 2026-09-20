# ----coding   : utf-8  ----
# @Time         : 2026/6/19 11:43
# @Author       : RisingInsight
# @File         : FSFNet_pretrain4.py
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
from torch.optim import lr_scheduler
from models.wavelet_transform import MultiLevelWaveletTransform
import torch.nn.functional as F

def calc_tokenized_size(image_shape, token_size):
    if image_shape[1] % token_size[0] != 0:
        raise ValueError("Token width %d does not divide image width %d" % (token_size[0], image_shape[1]))
    if image_shape[2] % token_size[1] != 0:
        raise ValueError("Token height %d does not divide image height %d" % (token_size[1], image_shape[2]))
    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1])

class ImagePatchRandomMasking(nn.Module):
    def __init__(self, patch_size, fraction=0.4):
        super().__init__()
        self._patch_size = patch_size
        self._fraction = fraction

    def forward(self, image):
        N_h, N_w = calc_tokenized_size(image.shape[1:], (self._patch_size, self._patch_size))
        mask = (torch.rand((image.shape[0], 1, N_h, N_w)) > self._fraction)
        mask = mask.repeat_interleave(self._patch_size, dim=2)
        mask = mask.repeat_interleave(self._patch_size, dim=3)
        return mask.to(image.device) * image, mask

class FSFNetPT(BaseModel):
    def name(self):
        return 'FSFNetPT'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.epoch = 0
        self.opt = opt
        self.use_wavelet = opt.use_wavelet

        self.input_A = torch.zeros((nb, opt.input_nc, size, size)).cuda(opt.gpu_ids[0])
        self.input_B = torch.zeros((nb, opt.output_nc, size, size)).cuda(opt.gpu_ids[0])

        self.netG_AB = FSFNetModel(opt).cuda(opt.gpu_ids[0])

        self.masking = ImagePatchRandomMasking(patch_size=opt.patch_size, fraction=opt.fraction)

        self.criterionIdt = torch.nn.L1Loss()
        self.criterionRes = torch.nn.MSELoss()

        self.model_names = ['G_AB']

        if self.use_wavelet:
            self.wavelet_transform = MultiLevelWaveletTransform(opt.input_nc, opt.gpu_ids[0], levels=2)

        if self.isTrain:
            self.init_weights()
            self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netG_AB.parameters()),
                                                 lr=opt.batchSize * 5e-3 / 512,
                                                 betas=(0.9, 0.99), weight_decay=0.05)
            self.scheduler_G = self.get_scheduler(self.optimizer_G, opt)

        if opt.continue_train:
            self.load_checkpoint(opt.which_epoch)

    def load_checkpoint(self, epoch_label):
        if epoch_label == 'latest':
            checkpoint_path = os.path.join(self.save_dir, 'pretrain_latest_checkpoint.pth')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'{epoch_label}_checkpoint.pth')

        if os.path.isfile(checkpoint_path):
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage.cuda(self.opt.gpu_ids[0]),
                                    weights_only=False)
            model_state_dict = {k: v.clone() for k, v in checkpoint['model_state_dict'].items()}
            self.netG_AB.load_state_dict(model_state_dict)
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.epoch = checkpoint['epoch']
            print(f'Resumed from epoch {self.epoch}')
        else:
            print(f'No checkpoint found at {checkpoint_path}, starting from scratch.')

    def save_checkpoint(self, epoch_label):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.netG_AB.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'opt': self.opt
        }
        if epoch_label == 'latest':
            save_path = os.path.join(self.save_dir, 'pretrain_latest_checkpoint.pth')
        else:
            save_path = os.path.join(self.save_dir, f'{epoch_label}_checkpoint.pth')
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved to {save_path}')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = self.input_A
        self.real_B = self.input_B

        self.mask_A, self.mask_for_A = self.masking(self.real_A)
        self.mask_B, self.mask_for_B = self.masking(self.real_B)

        self.rec_A, self.rec_A_freq = self.netG_AB(self.mask_A)
        self.rec_B, self.rec_B_freq = self.netG_AB(self.mask_B)

        if self.use_wavelet:
            self.rec_A_wavelets = self.wavelet_transform(self.rec_A)
            self.real_A_wavelets = self.wavelet_transform(self.real_A)
            self.rec_B_wavelets = self.wavelet_transform(self.rec_B)
            self.real_B_wavelets = self.wavelet_transform(self.real_B)

    def backward_G(self):
        loss_A = self.criterionRes(self.rec_A, self.real_A)
        loss_B = self.criterionRes(self.rec_B, self.real_B)

        loss_A_aux = self.criterionRes(self.rec_A_freq, self.real_A)*self.opt.lambda_aux
        loss_B_aux = self.criterionRes(self.rec_B_freq, self.real_B)*self.opt.lambda_aux

        if self.use_wavelet:
            lambda_low = self.opt.lambda_wavelet * 1.5
            lambda_high = self.opt.lambda_wavelet * 1.0
            loss_A_lf_total = 0
            loss_A_hf_total = 0
            loss_B_lf_total = 0
            loss_B_hf_total = 0

            for lvl, ((rec_ll, rec_lh, rec_hl, rec_hh), (real_ll, real_lh, real_hl, real_hh)) in enumerate(
                    zip(self.rec_A_wavelets, self.real_A_wavelets)):
                loss_A_lf = self.criterionRes(rec_ll, real_ll) * lambda_low
                loss_A_hf = (self.criterionRes(rec_lh, real_lh) +
                             self.criterionRes(rec_hl, real_hl) +
                             self.criterionRes(rec_hh, real_hh)) * lambda_high
                loss_A_lf_total += loss_A_lf
                loss_A_hf_total += loss_A_hf

            for lvl, ((rec_ll, rec_lh, rec_hl, rec_hh), (real_ll, real_lh, real_hl, real_hh)) in enumerate(
                    zip(self.rec_B_wavelets, self.real_B_wavelets)):
                loss_B_lf = self.criterionRes(rec_ll, real_ll) * lambda_low
                loss_B_hf = (self.criterionRes(rec_lh, real_lh) +
                             self.criterionRes(rec_hl, real_hl) +
                             self.criterionRes(rec_hh, real_hh)) * lambda_high
                loss_B_lf_total += loss_B_lf
                loss_B_hf_total += loss_B_hf

            loss = loss_A + loss_B + loss_A_lf_total + loss_A_hf_total + loss_B_lf_total + loss_B_hf_total + loss_A_aux + loss_B_aux

            self.loss_A_lf = loss_A_lf_total.item()
            self.loss_A_hf = loss_A_hf_total.item()
            self.loss_B_lf = loss_B_lf_total.item()
            self.loss_B_hf = loss_B_hf_total.item()
        else:
            loss = loss_A + loss_B + loss_A_aux + loss_B_aux

        loss.backward()

        self.loss_A = loss_A.item()
        self.loss_B = loss_B.item()

    def get_current_errors(self):
        if self.use_wavelet:
            ret_errors = OrderedDict(
                [('loss_A', self.loss_A), ('loss_B', self.loss_B),
                 ('loss_A_lf', self.loss_A_lf), ('loss_A_hf', self.loss_A_hf),
                 ('loss_B_lf', self.loss_B_lf), ('loss_B_hf', self.loss_B_hf)])
        else:
            ret_errors = OrderedDict([('loss_A', self.loss_A), ('loss_B', self.loss_B)])
        return ret_errors

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.opt.gpu_ids)
        self.save_checkpoint(label)

    def get_image_paths(self):
        return self.image_paths

    def test(self):
        self.real_A = self.input_A
        self.real_B = self.input_B
        self.mask_A, self.mask_for_A = self.masking(self.real_A)
        self.mask_B, self.mask_for_B = self.masking(self.real_B)
        self.rec_A = self.netG_AB(self.mask_A)
        self.rec_B = self.netG_AB(self.mask_B)

    def get_L1_error(self):
        loss = self.criterionRes(self.rec_A, self.real_A).item()
        loss += self.criterionRes(self.rec_B, self.real_B).item()
        return loss

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.detach().clone())
        real_B = util.tensor2im(self.real_B.detach().clone())
        mask_A = util.tensor2im(self.mask_A.detach().clone())
        mask_B = util.tensor2im(self.mask_B.detach().clone())
        rec_A = util.tensor2im(self.rec_A.detach().clone())
        rec_B = util.tensor2im(self.rec_B.detach().clone())

        if self.use_wavelet:
            real_A_ll = util.tensor2im(self.real_A_wavelets[0][0].detach().clone())
            real_A_lh = util.tensor2im(self.real_A_wavelets[0][1].detach().clone())
            real_A_hl = util.tensor2im(self.real_A_wavelets[0][2].detach().clone())
            real_A_hh = util.tensor2im(self.real_A_wavelets[0][3].detach().clone())
            real_B_ll = util.tensor2im(self.real_B_wavelets[0][0].detach().clone())
            real_B_lh = util.tensor2im(self.real_B_wavelets[0][1].detach().clone())
            real_B_hl = util.tensor2im(self.real_B_wavelets[0][2].detach().clone())
            real_B_hh = util.tensor2im(self.real_B_wavelets[0][3].detach().clone())

            rec_A_ll = util.tensor2im(self.rec_A_wavelets[0][0].detach().clone())
            rec_A_lh = util.tensor2im(self.rec_A_wavelets[0][1].detach().clone())
            rec_A_hl = util.tensor2im(self.rec_A_wavelets[0][2].detach().clone())
            rec_A_hh = util.tensor2im(self.rec_A_wavelets[0][3].detach().clone())
            rec_B_ll = util.tensor2im(self.rec_B_wavelets[0][0].detach().clone())
            rec_B_lh = util.tensor2im(self.rec_B_wavelets[0][1].detach().clone())
            rec_B_hl = util.tensor2im(self.rec_B_wavelets[0][2].detach().clone())
            rec_B_hh = util.tensor2im(self.rec_B_wavelets[0][3].detach().clone())

            ret_visuals = OrderedDict([
                ('real_A', real_A), ('real_B', real_B),
                ('mask_A', mask_A), ('mask_B', mask_B),
                ('rec_A', rec_A), ('rec_B', rec_B),
                ('real_A_ll', real_A_ll), ('real_A_lh', real_A_lh), ('real_A_hl', real_A_hl), ('real_A_hh', real_A_hh),
                ('real_B_ll', real_B_ll), ('real_B_lh', real_B_lh), ('real_B_hl', real_B_hl), ('real_B_hh', real_B_hh),
                ('rec_A_ll', rec_A_ll), ('rec_A_lh', rec_A_lh), ('rec_A_hl', rec_A_hl), ('rec_A_hh', rec_A_hh),
                ('rec_B_ll', rec_B_ll), ('rec_B_lh', rec_B_lh), ('rec_B_hl', rec_B_hl), ('rec_B_hh', rec_B_hh)
            ])
        else:
            ret_visuals = OrderedDict([
                ('real_A', real_A), ('real_B', real_B),
                ('mask_A', mask_A), ('mask_B', mask_B),
                ('rec_A', rec_A), ('rec_B', rec_B)
            ])
        return ret_visuals

    def update_learning_rate(self):
        self.scheduler_G.step()
        lr = self.optimizer_G.param_groups[0]['lr']
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

    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'CosineAnnealingWarmRestarts':
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1,
                                                            eta_min=opt.batchSize * 5e-8 / 512)
