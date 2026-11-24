from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=10,
                                 help='frequency of showing training results on screen')            # 在屏幕上显示训练结果的频率（每10个iteration显示一次）
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain '
                                      'number of images per row.')                                  # 在Visdom中显示图像时每行的图像数量
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')               # 保存训练结果到HTML文件的频率
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')           # 在控制台打印训练结果的频率
        self.parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')                     # 保存最新模型的频率（每5000次迭代）
        self.parser.add_argument('--save_epoch_freq', type=int, default=100,
                                 help='frequency of saving checkpoints at the end of epochs')       # 按epoch保存模型的频率（每100个epoch）
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')                   # 是否继续训练：加载最新模型继续训练
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, '
                                      '<epoch_count>+<save_latest_freq>, ...')                      # 起始epoch计数
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')        # 当前阶段：训练
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')      # 加载哪个epoch的模型：latest表示最新
        self.parser.add_argument('--epochs_warmup', type=int, default=100, help='# of epoch to  linearly warm up lr')                       # 学习率预热epoch数：前100个epoch线性增加学习率，有助于训练初期稳定收敛
        self.parser.add_argument('--epochs_anneal', type=int, default=100, help='# of epoch to linearly decay learning rate to zero')       # 学习率退火epoch数：后100个epoch线性衰减学习率到0，帮助模型精细调优
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')     # Adam优化器的初始学习率
        self.parser.add_argument('--gradient_penalty', default=True, help='use gradient_penality')              # 是否使用梯度惩罚（WGAN-GP等对抗训练技巧）
        self.parser.add_argument('--lambda_gp', type=float, default=0.1 / 100 ** 2, help='weight for gradient penalty')     # 梯度惩罚的权重系数
        self.parser.add_argument('--constant', type=float, default=100, help='constant for gradient penalty')   # 梯度惩罚中的常数项
        self.parser.add_argument('--gan_mode', default='lsgan', help='lsgan loss ')                             # GAN损失类型：最小二乘GAN（比原始GAN更稳定）
        self.parser.add_argument('--lambda_AB', type=float, default=10.0, help='weight for gc loss')            # GC损失（生成器损失）的权重
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')    # 循环一致性损失 A->B->A 的权重
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')    # 循环一致性损失 B->A->B 的权重
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')       # 图像缓冲区大小：存储之前生成的图像用于判别器训练
        # self.parser.add_argument('--no_html', action='store_true',
        #                          help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='linear',
                                 help='learning rate policy: linear ')                          # 学习率调度策略：线性变化
        self.parser.add_argument('--geometry', type=str, default='rot', help='pre-defined geometry transformation '
                                                                             'function: rot|vf|hf')     # 几何变换类型：旋转（rot）、垂直翻转（vf）、水平翻转（hf），数据增强


        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')            # 学习率衰减的迭代间隔（每50次迭代乘以gamma）
        self.parser.add_argument('--identity', type=float, default=0.5,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling '
                                      'the weight of the identity mapping loss. For example, if the weight of the '
                                      'identity loss should be 10 times smaller than the weight of the reconstruction '
                                      'loss, please set optidentity = 0.1')                                                 # 身份映射损失的权重
        self.parser.add_argument('--lambda_gc', type=float, default=2.0, help='trade-off parameter for Gc and idt')         # GC损失和身份损失的权衡参数
        self.parser.add_argument('--lambda_G', type=float, default=1.0, help='trade-off parameter for G, gc, and idt')      # 生成器总损失的权衡参数

        self.isTrain = True
