# --*--coding   : uft-8  --*--
# @Time         : 2026/2/22 16:17
# @Author       : RisingInsight
# @File         : test_options2.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load?')
        self.parser.add_argument('--geometry', type=str, default='rot', help='pre-defined geometry transformation function: rot|vf')
        # 测试专用预处理方式，默认直接缩放至 fineSize（避免随机裁剪）
        self.parser.add_argument('--test_preprocess', type=str, default='resize',
                                 choices=['resize', 'resize_and_crop', 'crop', 'none'],
                                 help='preprocessing for test images: resize (scale to fineSize), resize_and_crop (scale then center crop), crop (random crop, not recommended), none (use original size)')
        self.isTrain = False

