"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

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
import time
import copy
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # ===================== Efficiency measurement (read-only observers) =====================
    # FLOPs / Params of the generator G (the ONLY network used at test time).
    # Inference Time averaged over the test set, excluding a few warmup iterations.
    # These only OBSERVE model.test(); they never change its behavior or its outputs.
    efficiency_log = []
    WARMUP = 5
    time_total, time_count = 0.0, 0
    measured_flops = False
    # =======================================================================================

    total = len(dataset)
    inferred_count = 0
    print('=' * 64)
    print('=== AECM Inference ===')
    print('Test set size : %d images (batch_size=1, ONE image at a time)' % total)
    print('num_test cap  : %d  -> will infer every available test image' % opt.num_test)
    print('=' * 64)

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader

        # ---- one-shot FLOPs / Params measurement (ISOLATED clone; live model untouched) ----
        if not measured_flops:
            measured_flops = True
            try:
                from thop import profile
                # 关键修复：profile 一个隔离的 deepcopy 副本，绝不动被 DataParallel 包裹的活模型，
                # 否则 thop 的内部前向会把 netG.module 的设备位置带乱，导致后续 model.test() device mismatch。
                inner = model.netG.module if hasattr(model.netG, 'module') else model.netG
                g_clone = copy.deepcopy(inner).to('cpu').eval()
                in_shape = tuple(model.real_A.shape)            # (1, 3, H, W)
                dummy = torch.randn(in_shape)                   # CPU dummy；FLOPs 计数与设备无关
                macs, n_params = profile(g_clone, (dummy,), verbose=False)
                flops_g = 2.0 * macs                            # FLOPs = 2 x MACs
                efficiency_log.append('[Efficiency] Generator G  FLOPs : %.6f G   (input %s)'
                                      % (flops_g / 1e9, str(in_shape)))
                efficiency_log.append('[Efficiency] Generator G  Params: %.6f M' % (n_params / 1e6))
                del g_clone, dummy
                torch.cuda.empty_cache()
            except ImportError:
                efficiency_log.append('[Efficiency] thop not installed -> skip FLOPs/Params. Run: pip install thop')
            except Exception as e:
                efficiency_log.append('[Efficiency] FLOPs measurement failed: %s' % str(e))
            torch.cuda.synchronize()   # flush the dummy forward so it does not leak into timing

        # ---- timed inference (model.test() is called exactly as before) ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.test()           # run inference (unchanged)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= WARMUP:
            time_total += (t1 - t0)
            time_count += 1

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        inferred_count += 1
        print('Inferring image %04d / %04d ... %s' % (inferred_count, total, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML
    print('=' * 64)
    print('=== Done: inferred %d / %d test images -> %s ===' % (inferred_count, total, web_dir))
    if inferred_count < total:
        print('WARNING: only %d/%d test images were inferred (num_test=%d). '
              'Pass --num_test %d to infer the FULL test set.' % (inferred_count, total, opt.num_test, total))
    print('=' * 64)

    # ===================== Efficiency summary (print + persist) =====================
    if time_count > 0:
        avg_ms = (time_total / time_count) * 1000.0
        efficiency_log.append('[Efficiency] Inference Time: %.6f ms/image   (avg over %d images, excl. %d warmup)'
                              % (avg_ms, time_count, WARMUP))
    else:
        efficiency_log.append('[Efficiency] Inference Time: not measured (need > %d images)' % WARMUP)

    print('-' * 64)
    for _line in efficiency_log:
        print(_line)
    print('-' * 64)

    try:
        eff_path = os.path.join(web_dir, 'efficiency.txt')
        with open(eff_path, 'w') as _f:
            _f.write('\n'.join(efficiency_log) + '\n')
        print('Efficiency summary saved to', eff_path)
    except Exception as e:
        print('[Efficiency] failed to save efficiency.txt: %s' % str(e))
