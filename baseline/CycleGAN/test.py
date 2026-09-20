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
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


# --- efficiency measurement (shared across baselines; see baseline/scripts/efficiency.py) ---
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


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    total = len(dataset)
    inferred_count = 0
    print('=' * 64)
    print('=== CycleGAN Inference ===')
    print(f'Test set size : {total} images (batch_size=1, ONE image at a time)')
    print(f'num_test cap  : {opt.num_test}  -> will infer min(num_test, total) images')
    print('=' * 64)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    meter = EfficiencyMeter()  # [Efficiency] FLOPs / Inference Time observer

    # create a website
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        meter.measure_flops(get_generator(model), model.real_A)  # [Efficiency] one-shot, safe each iter
        meter.time_inference(model.test, i)  # [Efficiency] timed inference (replaces direct call)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        inferred_count += 1
        if i % 5 == 0:
            print(f'Inferring image {inferred_count:04d} / {total:04d} ... {img_path}')
    webpage.save()  # save the HTML
    meter.report(web_dir)  # [Efficiency] print + save efficiency.txt
    print('=' * 64)
    print(f'=== Done: inferred {inferred_count} / {total} test images -> {web_dir} ===')
    if inferred_count < total:
        print(f'WARNING: only {inferred_count}/{total} test images were inferred (num_test={opt.num_test}). '
              f'Pass --num_test {total} to infer the FULL test set.')
    print('=' * 64)
