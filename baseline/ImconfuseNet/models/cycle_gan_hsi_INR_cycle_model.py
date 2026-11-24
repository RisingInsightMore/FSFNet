import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_attention as networks
from torch.autograd import Variable
import scipy.io as sio
from . import liif

class CycleGANhsiINRCycleModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.sr_H = opt.sr_H
        self.sr_W = opt.sr_W
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L1_A_rgb', 'L1_A_infrared', 'L1_B_rgb', 'L1_B_infrared', 'D_A11', 'D_A12', 'D_A21', 'D_A22', 'G_A11', 'G_A12', 'G_A21', 'G_A22', 'cycle_A', 'idt_A', 'cycle_B', 'idt_B', 'smooth_c', 'smooth_h']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_A_infrared', 'fake_A_rgb', 'rec_A_rgb', 'fake_A_hsi']
        visual_names_B = ['real_B', 'fake_B_infrared', 'fake_B_rgb', 'rec_B_infrared', 'fake_B_hsi']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A1', 'D_A2']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
            #self.model_names = ['Pre', 'G_A', 'G_B', 'H']

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_LIIF(n_colors = 3, encoder_spec = opt.encoder_spec, imnet_spec = opt.imnet_spec, inp_size = opt.image_size, gpu_ids = self.gpu_ids)
        self.netG_B = networks.define_LIIF(n_colors = 1, encoder_spec = opt.encoder_spec, imnet_spec = opt.imnet_spec, inp_size = opt.image_size, gpu_ids = self.gpu_ids)
        
        
        self.srf_thermal = sio.loadmat('SRF_thermal.mat')
        self.srf_thermal = torch.from_numpy(self.srf_thermal['SRF'])
        self.srf_thermal = self.srf_thermal[:,10:20]
        self.srf_thermal = self.srf_thermal/torch.sum(self.srf_thermal)
        #self.srf_thermal = torch.Tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        self.srf_thermal = self.srf_thermal.type(torch.cuda.FloatTensor)
        self.SRF = sio.loadmat('P_N_V2.mat')
        self.SRF = torch.from_numpy(self.SRF['P_20N'])
        self.SRF = self.SRF[:,10:25]
        self.SRF = self.SRF/torch.sum(self.SRF)
        self.SRF = self.SRF.type(torch.cuda.FloatTensor)
        self.sr_H = opt.sr_H
        self.sr_W = opt.sr_W

        self.criterionL1 = torch.nn.L1Loss()

        if self.isTrain:  # define discriminators
            self.netD_A1 = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_A2 = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A1_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A2_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B1_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B2_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1Loss = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A1.parameters(), self.netD_A2.parameters()), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hr_coord = input['coord'].to(self.device)
        self.cell = input['cell'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_hsi = self.netG_A(self.real_A, self.hr_coord, self.cell, self.sr_H, self.sr_W)  # G_A(A)
        self.fake_B_hsi = self.fake_B_hsi.view(self.fake_B_hsi.shape[0],self.sr_H,self.sr_W,-1)
        self.fake_B_infrared = torch.matmul(self.fake_B_hsi[:,:,:,15:], self.srf_thermal.T)
        self.fake_B_infrared = self.fake_B_infrared.permute(0,3,1,2)
        self.fake_B_rgb = torch.matmul(self.fake_B_hsi[:,:,:,0:15], self.SRF.T)
        self.fake_B_rgb = self.fake_B_rgb.permute(0,3,1,2) 

        self.rec_A_hsi = self.netG_B(self.fake_B_infrared, self.hr_coord, self.cell, self.sr_H, self.sr_W)   # G_B(G_A(A))
        self.rec_A_hsi = self.rec_A_hsi.view(self.rec_A_hsi.shape[0],self.sr_H,self.sr_W,-1)
        self.rec_A_rgb = torch.matmul(self.rec_A_hsi[:,:,:,0:15], self.SRF.T)
        self.rec_A_rgb = self.rec_A_rgb.permute(0,3,1,2) 
        self.rec_A_infrared = torch.matmul(self.rec_A_hsi[:,:,:,15:], self.srf_thermal.T)
        self.rec_A_infrared = self.rec_A_infrared.permute(0,3,1,2)

        self.fake_A_hsi = self.netG_B(self.real_B, self.hr_coord, self.cell, self.sr_H, self.sr_W)  # G_B(B)
        self.fake_A_hsi = self.fake_A_hsi.view(self.fake_A_hsi.shape[0],self.sr_H,self.sr_W,-1)
        self.fake_A_infrared = torch.matmul(self.fake_A_hsi[:,:,:,15:], self.srf_thermal.T)
        self.fake_A_infrared = self.fake_A_infrared.permute(0,3,1,2)
        self.fake_A_rgb = torch.matmul(self.fake_A_hsi[:,:,:,0:15], self.SRF.T)
        self.fake_A_rgb = self.fake_A_rgb.permute(0,3,1,2) 
        
        self.rec_B_hsi = self.netG_A(self.fake_A_rgb, self.hr_coord, self.cell, self.sr_H, self.sr_W)   # G_A(G_B(B))
        self.rec_B_hsi = self.rec_B_hsi.view(self.rec_B_hsi.shape[0],self.sr_H,self.sr_W,-1)
        self.rec_B_infrared = torch.matmul(self.rec_B_hsi[:,:,:,15:], self.srf_thermal.T)
        self.rec_B_infrared = self.rec_B_infrared.permute(0,3,1,2)
        self.rec_B_rgb = torch.matmul(self.rec_B_hsi[:,:,:,0:15], self.SRF.T)
        self.rec_B_rgb = self.rec_B_rgb.permute(0,3,1,2)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A1(self):
        """Calculate GAN loss for discriminator D_A1"""
        fake_B_rgb = self.fake_A1_pool.query(self.fake_B_rgb)
        fake_A_rgb = self.fake_A2_pool.query(self.fake_A_rgb)
        self.loss_D_A11 = self.backward_D_basic(self.netD_A1, self.real_A, fake_B_rgb)
        self.loss_D_A12 = self.backward_D_basic(self.netD_A1, self.real_A, fake_A_rgb)

    def backward_D_A2(self):
        """Calculate GAN loss for discriminator D_A2"""
        fake_B_infrared = self.fake_B1_pool.query(self.fake_B_infrared)
        fake_A_infrared = self.fake_B2_pool.query(self.fake_A_infrared)
        self.loss_D_A21 = self.backward_D_basic(self.netD_A2, self.real_B, fake_B_infrared)
        self.loss_D_A22 = self.backward_D_basic(self.netD_A2, self.real_B, fake_A_infrared)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_L1_A_rgb = self.criterionL1Loss(self.real_A, self.fake_A_rgb) * lambda_A 
        self.loss_L1_A_infrared = self.criterionL1Loss(self.real_B, self.fake_A_infrared) * lambda_A 
        self.loss_L1_B_rgb = self.criterionL1Loss(self.real_A, self.fake_B_rgb) * lambda_A  
        self.loss_L1_B_infrared = self.criterionL1Loss(self.real_B, self.fake_B_infrared) * lambda_A  
        # GAN loss D_A(G_A(A))
        self.loss_G_A11 = self.criterionGAN(self.netD_A1(self.fake_B_rgb), True) 
        self.loss_G_A21 = self.criterionGAN(self.netD_A2(self.fake_B_infrared), True) 
        # GAN loss D_B(G_B(B))
        self.loss_G_A12 = self.criterionGAN(self.netD_A1(self.fake_A_rgb), True) 
        self.loss_G_A22 = self.criterionGAN(self.netD_A2(self.fake_A_infrared), True) 
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A_rgb, self.real_A) * lambda_A 
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B_infrared, self.real_B) * lambda_B 
        [B,C,H,W] = self.fake_B_hsi.shape
        fea1 = self.fake_B_hsi.reshape([B,C,H*W])
        input_1_c1 = fea1[:,1:,:]
        input_2_c1 = fea1[:,:-1,:]
        forder_c1 = input_1_c1 - input_2_c1
        #print(forder.size())
        self.loss_smooth_c1 = torch.mean(forder_c1**2)*5
        input_1_h1 = fea1[:,:,1:]
        input_2_h1 = fea1[:,:,:-1]
        forder_h1 = input_1_h1 - input_2_h1
        #print(forder.size())
        self.loss_smooth_h1 = torch.mean(forder_h1**2)*5

        fea2 = self.fake_A_hsi.reshape([B,C,H*W])
        input_1_c2 = fea2[:,1:,:]
        input_2_c2 = fea2[:,:-1,:]
        forder_c2 = input_1_c2 - input_2_c2
        #print(forder.size())
        self.loss_smooth_c2 = torch.mean(forder_c2**2)*5
        input_1_h2 = fea2[:,:,1:]
        input_2_h2 = fea2[:,:,:-1]
        forder_h2 = input_1_h2 - input_2_h2
        #print(forder.size())
        self.loss_smooth_h2 = torch.mean(forder_h2**2)*5

        self.loss_smooth_c = self.loss_smooth_c1 + self.loss_smooth_c2
        self.loss_smooth_h = self.loss_smooth_h1 + self.loss_smooth_h2
        # combined loss and calculate gradients

        self.loss_G = (
                      self.loss_L1_A_rgb + self.loss_L1_A_infrared + 
                      self.loss_L1_B_rgb + self.loss_L1_B_infrared + 
                      self.loss_G_A11 + self.loss_G_A12 + 
                      self.loss_G_A21 + self.loss_G_A22 + 
                      self.loss_cycle_A + self.loss_cycle_B + 
                      self.loss_idt_A + self.loss_idt_B + 
                      self.loss_smooth_c + self.loss_smooth_h)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # D_A and D_B
        self.set_requires_grad([self.netD_A1, self.netD_A2], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A1()      # calculate gradients for D_A
        self.backward_D_A2()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        
        # G_A and G_B
        self.set_requires_grad([self.netD_A1, self.netD_A2], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

