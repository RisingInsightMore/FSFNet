import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import pdb

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
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    meter = EfficiencyMeter()
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        model.set_input(data)
        meter.measure_flops(get_generator(model), model.input_A)   # USTNet: input_A set in set_input; real_A only assigned inside test()
        meter.time_inference(model.test, i)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
    meter.report(web_dir)




