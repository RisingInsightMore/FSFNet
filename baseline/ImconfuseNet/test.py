"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

import os as _os, sys as _sys
def _ensure_scripts_on_path():
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(6):
        _cand = _os.path.join(_here, 'scripts')
        if _os.path.isfile(_os.path.join(_cand, 'efficiency.py')):
            if _cand not in _sys.path:
                _sys.path.insert(0, _cand)
            return
        _here = _os.path.dirname(_here)
_ensure_scripts_on_path()
from efficiency import EfficiencyMeter, get_generator

import torch
from models.utils import make_coord


class _LIIFWrapper(torch.nn.Module):
    """Adapter so thop can profile the LIIF generator with a single tensor input.

    LIIF.forward requires (inp, coord, cell, height, weight); this wraps it so
    that ``profile(g_clone, (dummy,))`` works like the other baselines' generators.
    coord/cell are built with the SAME formula as data/unaligned_dataset.py.
    """
    def __init__(self, generator, sr_H, sr_W, channels):
        super().__init__()
        self.net = generator
        self.sr_H = sr_H
        self.sr_W = sr_W
        self.channels = channels

    def forward(self, inp):
        device = inp.device
        coord = make_coord([self.sr_H, self.sr_W, self.channels]).to(device)
        coord = coord.unsqueeze(0).expand(inp.shape[0], -1, -1).contiguous()
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / self.sr_H
        cell[:, :, 1] *= 2 / self.sr_W
        cell[:, :, 2] *= 2 / self.channels
        return self.net(inp, coord, cell, self.sr_H, self.sr_W)


def main():
    opt = TestOptions().parse()  # get test options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    meter = EfficiencyMeter()

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        # LIIF generator needs coord/cell/height/weight beyond the image tensor,
        # so wrap it for the single-tensor FLOPs profiler (efficiency.py unchanged).
        gen = get_generator(model)
        if hasattr(gen, 'module'):  # unwrap DataParallel / DistributedDataParallel
            gen = gen.module
        meter.measure_flops(_LIIFWrapper(gen, opt.sr_H, opt.sr_W, opt.channels), model.real_A)
        meter.time_inference(model.test, i)           # run inference (timed)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
    meter.report(web_dir)


if __name__ == '__main__':
    main()
