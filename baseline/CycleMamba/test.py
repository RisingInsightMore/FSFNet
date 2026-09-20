import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util import html
import time
import torch
import warnings
warnings.filterwarnings('ignore')
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

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options 
    # hard-code some parameters for test   
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # visualizer = Visualizer(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options  
    total = len(dataset)
    inferred_count = 0
    print('=' * 64)
    print('=== CycleMamba Inference ===')
    print('Test set size : %d images (batch_size=1, ONE image at a time)' % total)
    print('num_test cap  : %d  -> will infer min(num_test, total) images' % opt.num_test)
    print('=' * 64)
    model = create_model(opt)      # create a model given opt.model and other options   
    model.setup(opt)               # regular setup: load and print networks; create schedulers  
    meter = EfficiencyMeter()  # [Efficiency] FLOPs / Inference Time observer

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleMamba')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        meter.measure_flops(get_generator(model), model.real_A)  # [Efficiency] one-shot, safe each iter
        meter.time_inference(model.test, i)  # [Efficiency] timed inference (replaces direct call)
        visuals = model.get_current_visuals()  # get image results          
        img_path = model.get_image_paths()     # get image paths  
        # print('img_path', img_path)
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        inferred_count += 1
        if i % 5 == 0:
            print('Inferring image %04d / %04d ... %s' % (inferred_count, total, img_path))
    webpage.save()  # save the HTML
    meter.report(web_dir)  # [Efficiency] print + save efficiency.txt
    print('=' * 64)
    print('=== Done: inferred %d / %d test images -> %s ===' % (inferred_count, total, web_dir))
    if inferred_count < total:
        print('WARNING: only %d/%d test images were inferred (num_test=%d). ' \
              'Pass --num_test %d to infer the FULL test set.' % (inferred_count, total, opt.num_test, total))
    print('=' * 64)
