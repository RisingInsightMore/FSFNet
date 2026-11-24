import networks
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DR_AVIT(nn.Module):
    # 与 baseline 中其它模型(BaseModel.print_networks)完全一致的子网络清单与输出格式，
    # 使 DRAVIT 训练/推理时的参数打印逐行可比。顺序：编码器 -> 生成器 -> 判别器。
    _PARAM_NETWORK_NAMES = (
        'enc_c', 'enc_a', 'gen',
        'disA', 'disA_gc', 'disB', 'disB_gc',
        'disA2', 'disA2_gc', 'disB2', 'disB2_gc',
        'disContent', 'disContent_gc',
    )

    def __init__(self, opts):
        super(DR_AVIT, self).__init__()

        # parameters
        lr = 0.0001
        lr_dcontent = lr / 2.5
        self.nz = 8
        self.half_size = opts.batch_size // 2
        self.fine_size = opts.crop_size
        self.gpu = opts.gpu
        self.direction = opts.direction
        self.geometry = opts.geometry
        # Discriminator A
        self.disA = networks.Dis_domain(opts.input_dim_a, norm=opts.dis_norm)
        self.disA_gc = networks.Dis_domain(opts.input_dim_a, norm=opts.dis_norm)
        # Discriminator B
        self.disB = networks.Dis_domain(opts.input_dim_b, norm=opts.dis_norm)
        self.disB_gc = networks.Dis_domain(opts.input_dim_b, norm=opts.dis_norm)

        # Discriminator A2
        self.disA2 = networks.Dis_domain(opts.input_dim_a, norm=opts.dis_norm)
        self.disA2_gc = networks.Dis_domain(opts.input_dim_a, norm=opts.dis_norm)
        # Discriminator B2
        self.disB2 = networks.Dis_domain(opts.input_dim_b, norm=opts.dis_norm)
        self.disB2_gc = networks.Dis_domain(opts.input_dim_b, norm=opts.dis_norm)

        # Discriminator semantic structure
        self.disContent = networks.Dis_semantic_structure()
        self.disContent_gc = networks.Dis_semantic_structure()

        # encoders
        self.enc_c = networks.E_semantic_structure(opts.input_dim_a, opts.input_dim_b)

        self.enc_a = networks.E_imaging_style(opts.input_dim_a, opts.input_dim_b, self.nz)

        # decoders
        self.gen = networks.G_cross_domain(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

        # optimizers
        # A
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disA_gc_opt = torch.optim.Adam(self.disA_gc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # B
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_gc_opt = torch.optim.Adam(self.disB_gc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        # A2
        self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disA2_gc_opt = torch.optim.Adam(self.disA2_gc.parameters(), lr=lr, betas=(0.5, 0.999),
                                             weight_decay=0.0001)
        # B2
        self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB2_gc_opt = torch.optim.Adam(self.disB2_gc.parameters(), lr=lr, betas=(0.5,
                                                                                       0.999), weight_decay=0.0001)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999),
                                               weight_decay=0.0001)
        self.disContent_gc_opt = torch.optim.Adam(self.disContent_gc.parameters(), lr=lr_dcontent, betas=(0.5, 0.999),
                                                  weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Set up the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

        # param
        self.lambda_gc = opts.lambda_gc

    def print_networks(self, verbose):
        """Print total number of parameters per submodule, in the SAME format as the
        other baselines' BaseModel.print_networks(), so DRAVIT's parameter output is
        directly comparable in the paper's efficiency/params table.

        Called once after the model is built, both in train.py and test.py.
        """
        print('---------- Networks initialized -------------')
        for name in self._PARAM_NETWORK_NAMES:
            net = getattr(self, name, None)
            if net is None:
                continue
            num_params = 0
            for p in net.parameters():
                num_params += int(p.numel())
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def initialize(self):
        self.disA.apply(networks.gaussian_weights_init)
        self.disA_gc.apply(networks.gaussian_weights_init)
        self.disB.apply(networks.gaussian_weights_init)
        self.disB_gc.apply(networks.gaussian_weights_init)
        self.disA2.apply(networks.gaussian_weights_init)
        self.disA2_gc.apply(networks.gaussian_weights_init)
        self.disB2.apply(networks.gaussian_weights_init)
        self.disB2_gc.apply(networks.gaussian_weights_init)
        self.disContent.apply(networks.gaussian_weights_init)
        self.disContent_gc.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)
        self.enc_c.apply(networks.gaussian_weights_init)
        self.enc_a.apply(networks.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
        self.disA_gc_sch = networks.get_scheduler(self.disA_gc_opt, opts, last_ep)
        self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
        self.disB_gc_sch = networks.get_scheduler(self.disB_gc_opt, opts, last_ep)
        self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
        self.disA2_gc_sch = networks.get_scheduler(self.disA2_gc_opt, opts, last_ep)
        self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
        self.disB2_gc_sch = networks.get_scheduler(self.disB2_gc_opt, opts, last_ep)
        self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
        self.disContent_gc_sch = networks.get_scheduler(self.disContent_gc_opt, opts, last_ep)
        self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disA_gc.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.disB_gc.cuda(self.gpu)
        self.disA2.cuda(self.gpu)
        self.disA2_gc.cuda(self.gpu)
        self.disB2.cuda(self.gpu)
        self.disB2_gc.cuda(self.gpu)
        self.disContent.cuda(self.gpu)
        self.disContent_gc.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def test_forward(self, image, a2b=True):
        self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
        if a2b:
            self.z_content = self.enc_c.forward_a(image)
            output = self.gen.forward_b(self.z_content, self.z_random)
        else:
            self.z_content = self.enc_c.forward_b(image)
            output = self.gen.forward_a(self.z_content, self.z_random)
        return output

    def test_forward_transfer(self, image_a, image_b, a2b=True):
        self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)

        self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
        if a2b:
            output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
        else:
            output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
        return output

    # rotation 90
    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.fine_size
        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.gpu)
        if direction == 0:
            tensor = torch.index_select(tensor, 3, inv_idx)
        else:
            tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def vf(self, tensor):
        size = self.fine_size
        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.gpu)
        tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def hf(self, tensor):
        size = self.fine_size
        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.gpu)
        tensor = torch.index_select(tensor, 3, inv_idx)  # hf
        return tensor

    def forward(self):
        # input images
        half_size = self.half_size
        real_A = self.input_A
        real_B = self.input_B

        self.real_A_encoded = real_A[0:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B_encoded = real_B[0:half_size]
        self.real_B_random = real_B[half_size:]

        # gc
        if self.geometry == 'gc':
            # clockwise rot direction = 0 else 1
            self.real_A_gc_encoded = self.rot90(self.real_A_encoded.clone(), self.direction)
            self.real_A_gc_random = self.rot90(self.real_A_random.clone(), self.direction)
            self.real_B_gc_encoded = self.rot90(self.real_B_encoded.clone(), self.direction)
            self.real_B_gc_random = self.rot90(self.real_B_random.clone(), self.direction)
        elif self.geometry == 'vf':
            self.real_A_gc_encoded = self.vf(self.real_A_encoded.clone())
            self.real_A_gc_random = self.vf(self.real_A_random.clone())
            self.real_B_gc_encoded = self.vf(self.real_B_encoded.clone())
            self.real_B_gc_random = self.vf(self.real_B_random.clone())
        else:
            self.real_A_gc_encoded = self.hf(self.real_A_encoded.clone())
            self.real_A_gc_random = self.hf(self.real_A_random.clone())
            self.real_B_gc_encoded = self.hf(self.real_B_encoded.clone())
            self.real_B_gc_random = self.hf(self.real_B_random.clone())

        # get encoded z_c
        self.z_content_a, self.z_content_b = checkpoint(self.enc_c.forward, self.real_A_encoded, self.real_B_encoded, use_reentrant=False)
        # get encoded z_gc_c
        self.z_content_gc_a, self.z_content_gc_b = checkpoint(self.enc_c.forward, self.real_A_gc_encoded, self.real_B_gc_encoded, use_reentrant=False)

        # get encoded z_a
        self.z_attr_a, self.z_attr_b = checkpoint(self.enc_a.forward, self.real_A_encoded, self.real_B_encoded, use_reentrant=False)
        # get encoded z_gc_a
        self.z_attr_gc_a, self.z_attr_gc_b = checkpoint(self.enc_a.forward, self.real_A_gc_encoded, self.real_B_gc_encoded, use_reentrant=False)

        # get random z_a
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

        # Stabilizer: keep latent codes bounded so the generator decoders
        # cannot overflow into NaN on a transient spike.  Values this large
        # are degenerate anyway and only hurt training.
        self.z_content_a = torch.clamp(self.z_content_a, -10.0, 10.0)
        self.z_content_b = torch.clamp(self.z_content_b, -10.0, 10.0)
        self.z_content_gc_a = torch.clamp(self.z_content_gc_a, -10.0, 10.0)
        self.z_content_gc_b = torch.clamp(self.z_content_gc_b, -10.0, 10.0)
        self.z_attr_a = torch.clamp(self.z_attr_a, -10.0, 10.0)
        self.z_attr_b = torch.clamp(self.z_attr_b, -10.0, 10.0)
        self.z_attr_gc_a = torch.clamp(self.z_attr_gc_a, -10.0, 10.0)
        self.z_attr_gc_b = torch.clamp(self.z_attr_gc_b, -10.0, 10.0)
        self.z_random = torch.clamp(self.z_random, -10.0, 10.0)

        # content
        input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b), 0)
        input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a), 0)
        # gc_content
        input_content_gc_forA = torch.cat((self.z_content_gc_b, self.z_content_gc_a, self.z_content_gc_b), 0)
        input_content_gc_forB = torch.cat((self.z_content_gc_a, self.z_content_gc_b, self.z_content_gc_a), 0)
        # attr
        input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random), 0)
        input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random), 0)
        # gc_attr
        input_attr_gc_forA = torch.cat((self.z_attr_gc_a, self.z_attr_gc_a, self.z_random), 0)
        input_attr_gc_forB = torch.cat((self.z_attr_gc_b, self.z_attr_gc_b, self.z_random), 0)

        # fake — use checkpoint to avoid inplace op graph corruption
        output_fakeA = checkpoint(self.gen.forward_a, input_content_forA, input_attr_forA, use_reentrant=False)
        output_fakeB = checkpoint(self.gen.forward_b, input_content_forB, input_attr_forB, use_reentrant=False)
        # fake_gc
        output_gc_fakeA = checkpoint(self.gen.forward_a, input_content_gc_forA, input_attr_gc_forA, use_reentrant=False)
        output_gc_fakeB = checkpoint(self.gen.forward_b, input_content_gc_forB, input_attr_gc_forB, use_reentrant=False)

        # translate
        self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA,
                                                                                    self.z_content_a.size(0), dim=0)
        self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB,
                                                                                    self.z_content_a.size(0), dim=0)
        # translate gc
        self.fake_A_gc_encoded, self.fake_AA_gc_encoded, self.fake_A_gc_random = torch.split(output_gc_fakeA,
                                                                                             self.z_content_gc_a.size(
                                                                                                 0), dim=0)
        self.fake_B_gc_encoded, self.fake_BB_gc_encoded, self.fake_B_gc_random = torch.split(output_gc_fakeB,
                                                                                             self.z_content_gc_a.size(
                                                                                                 0), dim=0)

        # for display
        self.image_display = torch.cat(
            (self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
             self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(),
             self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(),
             self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu()), dim=0)

        # for latent regression
        self.z_attr_random_a, self.z_attr_random_b = checkpoint(self.enc_a.forward, self.fake_A_random, self.fake_B_random, use_reentrant=False)
        # for latent regression gc
        self.z_attr_gc_random_a, self.z_attr_gc_random_b = checkpoint(self.enc_a.forward, self.fake_A_gc_random,
                                                                              self.fake_B_gc_random, use_reentrant=False)

    def forward_content(self):
        half_size = self.half_size
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        # print(self.real_A_encoded.device)
        self.real_A_gc_encoded = self.rot90(self.real_A_encoded.clone(), self.direction)
        # print(self.real_A_gc_encoded.device)
        self.real_B_gc_encoded = self.rot90(self.real_B_encoded.clone(), self.direction)
        # get encoded z_c
        self.z_content_a, self.z_content_b = checkpoint(self.enc_c.forward, self.real_A_encoded, self.real_B_encoded, use_reentrant=False)
        # get encoded z_gc_c
        self.z_content_gc_a, self.z_content_gc_b = checkpoint(self.enc_c.forward, self.real_A_gc_encoded, self.real_B_gc_encoded, use_reentrant=False)

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
            AB_gt = self.rot90(AB_gc.clone().detach(), 1)
            loss_gc = self.criterionL1(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 0)
            loss_gc += self.criterionL1(AB_gc, AB_gc_gt)
        else:
            AB_gt = self.rot90(AB_gc.clone().detach(), 0)
            loss_gc = self.criterionL1(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 1)
            loss_gc += self.criterionL1(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.lambda_gc
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.fine_size

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.gpu)

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionL1(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionL1(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.lambda_gc

        return loss_gc

    def get_gc_hf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.fine_size

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.gpu)

        AB_gt = torch.index_select(AB_gc.clone().detach(), 3, inv_idx)
        loss_gc = self.criterionL1(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 3, inv_idx)
        loss_gc += self.criterionL1(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.lambda_gc

        return loss_gc

    def _safe_d_step(self, opt, net, loss, name):
        """The .backward() for this discriminator has ALREADY run inside
        backward_D / backward_contentD.  If the resulting loss is non-finite
        (which happens when the generator emitted a NaN batch), we must NOT
        let that NaN gradient reach the weights -- discard it via zero_grad
        and skip the step.  Without this guard a single bad batch poisons a
        discriminator permanently; afterwards every iteration is NaN forever
        (this was the ep96 crash).  With the guard, the next clean batch
        resumes normally and neither G nor D ever becomes NaN."""
        if torch.isnan(loss) or torch.isinf(loss):
            print('[WARN] update_D: %s loss non-finite (%.6g); skipping step' % (name, float(loss)))
            opt.zero_grad()
            return 0.0
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        opt.step()
        return loss.item()

    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        # Require grad when updating D
        self.set_requires_grad([self.disContent, self.disContent_gc], True)
        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.disContent, self.z_content_a, self.z_content_b)
        self.disContent_loss = self._safe_d_step(self.disContent_opt, self.disContent, loss_D_Content, 'disContent')

        # update discontent_gc
        self.disContent_gc_opt.zero_grad()
        loss_D_gc_Content = self.backward_contentD(self.disContent_gc, self.z_content_gc_a, self.z_content_gc_b)
        self.disContent_gc_loss = self._safe_d_step(self.disContent_gc_opt, self.disContent_gc, loss_D_gc_Content, 'disContent_gc')

    def update_D(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward()

        # Require grad when updating D
        self.set_requires_grad(
            [self.disA, self.disA_gc, self.disB, self.disB_gc, self.disA2, self.disA2_gc, self.disB2, self.disB2_gc,
             self.disContent, self.disContent_gc],
            True)

        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
        self.disA_loss = self._safe_d_step(self.disA_opt, self.disA, loss_D1_A, 'disA')

        # update disA_gc
        self.disA_gc_opt.zero_grad()
        loss_D1_A_gc = self.backward_D(self.disA_gc, self.real_A_gc_encoded, self.fake_A_gc_encoded)
        self.disA_gc_loss = self._safe_d_step(self.disA_gc_opt, self.disA_gc, loss_D1_A_gc, 'disA_gc')

        # update disA2
        self.disA2_opt.zero_grad()
        loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
        self.disA2_loss = self._safe_d_step(self.disA2_opt, self.disA2, loss_D2_A, 'disA2')

        # update disA2_gc
        self.disA2_gc_opt.zero_grad()
        loss_D2_A_gc = self.backward_D(self.disA2_gc, self.real_A_gc_random, self.fake_A_gc_random)
        self.disA2_gc_loss = self._safe_d_step(self.disA2_gc_opt, self.disA2_gc, loss_D2_A_gc, 'disA2_gc')

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
        self.disB_loss = self._safe_d_step(self.disB_opt, self.disB, loss_D1_B, 'disB')

        # update disB_gc
        self.disB_gc_opt.zero_grad()
        loss_D1_gc_B = self.backward_D(self.disB_gc, self.real_B_gc_encoded, self.fake_B_gc_encoded)
        self.disB_gc_loss = self._safe_d_step(self.disB_gc_opt, self.disB_gc, loss_D1_gc_B, 'disB_gc')

        # update disB2
        self.disB2_opt.zero_grad()
        loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
        self.disB2_loss = self._safe_d_step(self.disB2_opt, self.disB2, loss_D2_B, 'disB2')
        # update disB2_gc
        self.disB2_gc_opt.zero_grad()
        loss_D2_B_gc = self.backward_D(self.disB2_gc, self.real_B_gc_random, self.fake_B_gc_random)
        self.disB2_gc_loss = self._safe_d_step(self.disB2_gc_opt, self.disB2_gc, loss_D2_B_gc, 'disB2_gc')

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.disContent, self.z_content_a, self.z_content_b)
        self.disContent_loss = self._safe_d_step(self.disContent_opt, self.disContent, loss_D_Content, 'disContent')

        # update discontent_gc
        self.disContent_gc_opt.zero_grad()
        loss_D_gc_Content = self.backward_contentD(self.disContent_gc, self.z_content_gc_a, self.z_content_gc_b)
        self.disContent_gc_loss = self._safe_d_step(self.disContent_gc_opt, self.disContent_gc, loss_D_gc_Content, 'disContent_gc')

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        all0 = torch.zeros_like(pred_fake).cuda(self.gpu)
        all1 = torch.ones_like(pred_real).cuda(self.gpu)
        # Use the *logits* variant: numerically stable (no explicit sigmoid -> no log(0) -> no NaN).
        # Mathematically identical to binary_cross_entropy(sigmoid(x), y).
        ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(pred_fake, all0)
        ad_true_loss = nn.functional.binary_cross_entropy_with_logits(pred_real, all1)
        loss_D = ad_true_loss + ad_fake_loss

        loss_D.backward()
        return loss_D

    def backward_contentD(self, ContentD, imageA, imageB):
        pred_fake = ContentD.forward(imageA.detach())
        pred_real = ContentD.forward(imageB.detach())
        all1 = torch.ones((pred_real.size(0))).cuda(self.gpu)
        all0 = torch.zeros((pred_fake.size(0))).cuda(self.gpu)
        ad_true_loss = nn.functional.binary_cross_entropy_with_logits(pred_real, all1)
        ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(pred_fake, all0)

        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_EG(self):
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        # Do not require grad when updating E G
        self.set_requires_grad(
            [self.disA, self.disB, self.disA_gc, self.disB_gc, self.disA2, self.disA2_gc, self.disB2, self.disB2_gc,
             self.disContent, self.disContent_gc],
            False)
        # update EG
        self.backward_EG()
        if getattr(self, '_eg_skip', False):
            # backward_EG detected a non-finite loss and skipped backward, so the
            # generator/encoder weights are untouched. Do not step them either.
            self.enc_c_opt.zero_grad()
            self.enc_a_opt.zero_grad()
            self.gen_opt.zero_grad()
            return
        # clip generator/encoder grads: 12 GAN losses + gc losses can spike G grads
        nn.utils.clip_grad_norm_(self.enc_c.parameters(), 5)
        nn.utils.clip_grad_norm_(self.enc_a.parameters(), 5)
        nn.utils.clip_grad_norm_(self.gen.parameters(), 5)
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_EG(self):

        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.disContent, self.z_content_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.disContent, self.z_content_b)

        # content gc Ladv for generator
        loss_G_GAN_gc_Acontent = self.backward_G_GAN_content(self.disContent_gc, self.z_content_gc_a)
        loss_G_GAN_gc_Bcontent = self.backward_G_GAN_content(self.disContent_gc, self.z_content_gc_b)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)
        loss_G_GAN_A2 = self.backward_G_GAN(self.fake_A_random, self.disA2)
        loss_G_GAN_B2 = self.backward_G_GAN(self.fake_B_random, self.disB2)
        # Ladv gc for generator
        loss_G_GAN_gc_A = self.backward_G_GAN(self.fake_A_gc_encoded, self.disA_gc)
        loss_G_GAN_gc_B = self.backward_G_GAN(self.fake_B_gc_encoded, self.disB_gc)
        loss_G_GAN_gc_A2 = self.backward_G_GAN(self.fake_A_gc_random, self.disA2_gc)
        loss_G_GAN_gc_B2 = self.backward_G_GAN(self.fake_B_gc_random, self.disB2_gc)

        # KL loss - z_a
        loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
        loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01
        # kl loss - z_gc_a
        loss_kl_za_gc_a = self._l2_regularize(self.z_attr_gc_a) * 0.01
        loss_kl_za_gc_b = self._l2_regularize(self.z_attr_gc_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01
        # kl loss - z_gc_c
        loss_kl_zc_gc_a = self._l2_regularize(self.z_content_gc_a) * 0.01
        loss_kl_zc_gc_b = self._l2_regularize(self.z_content_gc_b) * 0.01

        # cross cycle consistency loss
        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10
        loss_G_L1_gc_AA = self.criterionL1(self.fake_AA_gc_encoded, self.real_A_gc_encoded) * 10
        loss_G_L1_gc_BB = self.criterionL1(self.fake_BB_gc_encoded, self.real_B_gc_encoded) * 10

        # gc loss
        if self.geometry == 'gc':
            loss_Gc_A = self.get_gc_rot_loss(self.fake_A_encoded, self.fake_A_gc_encoded, self.direction)
            loss_Gc_A_random = self.get_gc_rot_loss(self.fake_A_random, self.fake_A_gc_random, self.direction)
            loss_Gc_B = self.get_gc_rot_loss(self.fake_B_encoded, self.fake_B_gc_encoded, self.direction)
            loss_Gc_B_random = self.get_gc_rot_loss(self.fake_B_random, self.fake_B_gc_random, self.direction)
        elif self.geometry == 'hf':
            loss_Gc_A = self.get_gc_hf_loss(self.fake_A_encoded, self.fake_A_gc_encoded)
            loss_Gc_A_random = self.get_gc_hf_loss(self.fake_A_random, self.fake_A_gc_random)
            loss_Gc_B = self.get_gc_hf_loss(self.fake_B_encoded, self.fake_B_gc_encoded)
            loss_Gc_B_random = self.get_gc_hf_loss(self.fake_B_random, self.fake_B_gc_random)
        else:
            loss_Gc_A = self.get_gc_vf_loss(self.fake_A_encoded, self.fake_A_gc_encoded)
            loss_Gc_A_random = self.get_gc_vf_loss(self.fake_A_random, self.fake_A_gc_random)
            loss_Gc_B = self.get_gc_vf_loss(self.fake_B_encoded, self.fake_B_gc_encoded)
            loss_Gc_B_random = self.get_gc_vf_loss(self.fake_B_random, self.fake_B_gc_random)

        # latent regression loss
        loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
        loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10
        # latent regression gc loss
        loss_z_L1_gc_a = torch.mean(torch.abs(self.z_attr_gc_random_a - self.z_random)) * 10
        loss_z_L1_gc_b = torch.mean(torch.abs(self.z_attr_gc_random_b - self.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_z_L1_gc_a + loss_z_L1_gc_b

        loss_G = loss_G_GAN_A + loss_G_GAN_B + loss_G_GAN_gc_A + loss_G_GAN_gc_B + loss_G_GAN_A2 + loss_G_GAN_B2 + loss_G_GAN_gc_A2 + loss_G_GAN_gc_B2 + \
                 loss_G_GAN_Acontent + loss_G_GAN_Bcontent + loss_G_GAN_gc_Acontent + loss_G_GAN_gc_Bcontent + \
                 loss_G_L1_AA + loss_G_L1_BB + loss_G_L1_gc_AA + loss_G_L1_gc_BB + \
                 loss_kl_zc_a + loss_kl_zc_b + loss_kl_zc_gc_a + loss_kl_zc_gc_b + \
                 loss_kl_za_a + loss_kl_za_b + loss_kl_za_gc_a + loss_kl_za_gc_b + \
                 loss_Gc_A + loss_Gc_A_random + loss_Gc_B + loss_Gc_B_random + \
                 loss_z_L1

        if torch.isnan(loss_G) or torch.isinf(loss_G):
            # A transient spike or a bad input sample produced a non-finite loss.
            # Skip backward/step entirely so a single bad batch can't poison the
            # whole model -- the next (clean) batch resumes from good weights.
            print('[WARN] backward_EG: loss_G non-finite (%.6g); skipping this update' % float(loss_G))
            self._eg_skip = True
            return
        self._eg_skip = False

        loss_G.backward()
        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_gc_a = loss_G_GAN_gc_A.item()
        self.gan_loss_gc_b = loss_G_GAN_gc_B.item()
        self.gan_loss_a2 = loss_G_GAN_A2.item()
        self.gan_loss_b2 = loss_G_GAN_B2.item()
        self.gan_loss_gc_a2 = loss_G_GAN_gc_A2.item()
        self.gan_loss_gc_b2 = loss_G_GAN_gc_B2.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.gan_loss_gc_acontent = loss_G_GAN_gc_Acontent.item()
        self.gan_loss_gc_bcontent = loss_G_GAN_gc_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.loss_kl_za_gc_a = loss_kl_za_gc_a.item()
        self.loss_kl_za_gc_b = loss_kl_za_gc_b.item()
        self.loss_kl_zc_gc_a = loss_kl_zc_gc_a.item()
        self.loss_kl_zc_gc_b = loss_kl_zc_gc_b.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.l1_recon_AA_gc_loss = loss_G_L1_gc_AA.item()
        self.l1_recon_BB_gc_loss = loss_G_L1_gc_BB.item()

        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        self.l1_recon_z_loss_gc_a = loss_z_L1_gc_a.item()
        self.l1_recon_z_loss_gc_b = loss_z_L1_gc_b.item()
        self.gc_loss_a = loss_Gc_A.item() + loss_Gc_A_random.item()
        self.gc_loss_b = loss_Gc_B.item() + loss_Gc_B_random.item()
        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, ContentD, data):
        out = ContentD.forward(data)
        all_half = 0.5 * torch.ones((out.size(0))).cuda(self.gpu)
        ad_loss = nn.functional.binary_cross_entropy_with_logits(out, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD=None):
        out_fake = netD.forward(fake)
        all_ones = torch.ones_like(out_fake).cuda(self.gpu)
        loss_G = nn.functional.binary_cross_entropy_with_logits(out_fake, all_ones)
        return loss_G

    def update_lr(self):
        self.disA_sch.step()
        self.disA_gc_sch.step()
        self.disB_sch.step()
        self.disB_gc_sch.step()
        self.disA2_sch.step()
        self.disA2_gc_sch.step()
        self.disB2_sch.step()
        self.disB2_gc_sch.step()
        self.disContent_sch.step()
        self.disContent_gc_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location='cpu')
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disA_gc.load_state_dict(checkpoint['disA_gc'])
            self.disA2.load_state_dict(checkpoint['disA2'])
            self.disA2_gc.load_state_dict(checkpoint['disA2_gc'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.disB_gc.load_state_dict(checkpoint['disB_gc'])
            self.disB2.load_state_dict(checkpoint['disB2'])
            self.disB2_gc.load_state_dict(checkpoint['disB2_gc'])
            self.disContent.load_state_dict(checkpoint['disContent'])
            self.disContent_gc.load_state_dict(checkpoint['disContent_gc'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disA_gc_opt.load_state_dict(checkpoint['disA_gc_opt'])
            self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
            self.disA2_gc_opt.load_state_dict(checkpoint['disA2_gc_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.disB_gc_opt.load_state_dict(checkpoint['disB_gc_opt'])
            self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
            self.disB2_gc_opt.load_state_dict(checkpoint['disB2_gc_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.disContent_gc_opt.load_state_dict(checkpoint['disContent_gc_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'disA': self.disA.state_dict(),
            'disA_gc': self.disA_gc.state_dict(),
            'disB': self.disB.state_dict(),
            'disB_gc': self.disB_gc.state_dict(),
            'disA2': self.disA2.state_dict(),
            'disA2_gc': self.disA2_gc.state_dict(),
            'disB2': self.disB2.state_dict(),
            'disB2_gc': self.disB2_gc.state_dict(),
            'disContent': self.disContent.state_dict(),
            'disContent_gc': self.disContent_gc.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'disA_gc_opt': self.disA_gc_opt.state_dict(),
            'disB_opt': self.disB_opt.state_dict(),
            'disB_gc_opt': self.disB_gc_opt.state_dict(),
            'disA2_opt': self.disA2_opt.state_dict(),
            'disA2_gc_opt': self.disA2_gc_opt.state_dict(),
            'disB2_opt': self.disB2_opt.state_dict(),
            'disB2_gc_opt': self.disB2_gc_opt.state_dict(),
            'disContent_opt': self.disContent_opt.state_dict(),
            'disContent_gc_opt': self.disContent_gc_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A_encoded).detach()
        images_b = self.normalize_image(self.real_B_encoded).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        images_a2 = self.normalize_image(self.fake_A_random).detach()
        images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        images_b2 = self.normalize_image(self.fake_B_random).detach()
        images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
        row1 = torch.cat(
            (images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::]), 3)
        row2 = torch.cat(
            (images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]
