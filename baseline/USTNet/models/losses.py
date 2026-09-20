# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan/base/LICENSE` for copyright attribution and LICENSE

import torch
from torch import nn


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(
            self, gan_mode, target_real_label=1.0, target_fake_label=0.0
    ):
        """ Initialize the GANLoss class. 初始化GANLoss类

        Parameters:
            gan_mode (str) -- the type of GAN objective. GAN目标函数的类型
                Choices: vanilla, lsgan, and wgangp.
            target_real_label (bool) -- label for a real image      真实图像的标签
            target_fake_label (bool) -- label of a fake image       生成图像的标签

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. Vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()

        # pylint: disable=not-callable，使用register_buffer将张量注册为模块的缓冲区（不参与梯度计算）
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()                # 最小二乘GAN使用MSE损失
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()      # 原始GAN使用带sigmoid的BCE损失
        elif gan_mode == 'wgan':
            self.loss = None                        # WGAN不使用传统的损失函数
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) -- tpyically the prediction from a discriminator                  判别器的预测输出
            target_is_real (bool) -- if the ground truth label is for real images or fake images  真实图像还是生成图像的标签

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
            用真实标签填充的标签张量，大小与输入相同
        """

        if target_is_real:
            target_tensor = self.real_label                 # 真实图像的标签张量
        else:
            target_tensor = self.fake_label                 # 生成图像的标签张量
        return target_tensor.expand_as(prediction)          # 将标签张量扩展为与prediction相同的形状

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
           根据判别器的输出和真实标签计算损失

        Parameters:
            prediction (tensor) -- tpyically the prediction output from a discriminator             通常是判别器的预测输出
            target_is_real (bool) -- if the ground truth label is for real images or fake images    真实图像还是生成图像的标签

        Returns:
            the calculated loss.        计算得到的损失值
        """
        # WGAN损失，对于真实图像最大化判别器输出，对于生成器图像最小化判别器输出
        if self.gan_mode == 'wgan':
            if target_is_real:
                return -prediction.mean()           # 真实图像，最大化判别器输出
            else:
                return prediction.mean()            # 生成图像，最小化判别器输出

        # lsgan和vanilla，获取目标张量并计算损失
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


# pylint: disable=too-many-arguments
# pylint: disable=redefined-builtin
def cal_gradient_penalty(
        netD, real_data, fake_data, device,
        type='mixed', constant=1.0, lambda_gp=10.0
):
    """计算梯度惩罚损失，用于WGAN-GP
    Calculate the gradient penalty loss, used in WGAN-GP

    source: https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network，判别器网络
        real_data (tensor array)    -- real images，真实图像
        fake_data (tensor array)    -- generated images from the generator，生成器生成的图像
        device (str)                -- torch device，torch设备
        type (str)                  -- if we mix real and fake data or not，是否混合真实和生成数据
            Choices: [real | fake | mixed].
        constant (float)            -- the constant used in formula: (||gradient||_2 - constant)^2，公式中使用的常数
        lambda_gp (float)           -- weight for this loss，此损失函数的权重

    Returns the gradient penalty loss，返回梯度惩罚损失
    """
    if lambda_gp == 0.0:
        return 0.0, None        # 如果权重为0，直接返回0

    # 根据类型选择插值点
    if type == 'real':
        interpolatesv = real_data       # 在真实数据上计算梯度惩罚
    elif type == 'fake':
        interpolatesv = fake_data       # 在生成数据上计算梯度惩罚
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device=device)        # 在真实和生成数据之间随机插值
        alpha = alpha.expand(
            real_data.shape[0], real_data.nelement() // real_data.shape[0]
        ).contiguous().view(*real_data.shape)                           # 扩展alpha到与real_data相同的形状

        # x_hat = alpha * real + (1-alpha) * fake
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)   # 线性插值
    else:
        raise NotImplementedError('{} not implemented'.format(type))

    interpolatesv.requires_grad_(True)          # 设置需要梯度
    disc_interpolates = netD(interpolatesv)     # 判别器对插值点的输出

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolatesv,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    gradients = gradients[0].view(real_data.size(0), -1)        # 重塑梯度形状

    # 计算梯度惩罚，(||gradient||_2 - 1)^2 的均值 * lambda_gp
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp

    return gradient_penalty, gradients
