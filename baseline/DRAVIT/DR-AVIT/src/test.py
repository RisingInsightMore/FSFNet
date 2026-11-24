import time

import torch
from options import TestOptions
from dataset import dataset_single
from model import DR_AVIT
from saver import save_imgs
import os
import copy


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
from efficiency import EfficiencyMeter


def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset_A = dataset_single(opts, 'A', opts.input_dim_a)
    dataset_B = dataset_single(opts, 'B', opts.input_dim_b)
    loader_A = torch.utils.data.DataLoader(dataset_A, batch_size=1, num_workers=opts.nThreads)
    loader_B = torch.utils.data.DataLoader(dataset_B, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = DR_AVIT(opts)

    # 打印各子网络参数量（与 baseline 其它模型格式一致，训练/推理各一次）
    model.print_networks(False)

    model.resume(opts.resume, train=False)
    model.setgpu(opts.gpu)
    model.eval()

    meter = EfficiencyMeter()

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    os.makedirs(result_dir, exist_ok=True)

    # test
    print('\n--- testing ---')
    for idx2, (img2, name) in enumerate(loader_B):
        print('{}/{}'.format(idx2, len(loader_B)))
        img = img2.cuda(opts.gpu)
        img = [img]
        name = [name[0].split('.')[0]]
        save_imgs(img, name, result_dir)

    for idx1, (img1, name) in enumerate(loader_A):
        print('{}/{}'.format(idx1, len(loader_A)))
        img1 = img1.cuda(opts.gpu)
        imgs = []
        names = []
        # ---- FLOPs of the composite generator (enc_c -> gen), measured ONCE, isolated on CPU ----
        # NOTE: thop.profile requires a REAL nn.Module (it calls .eval()/.training and
        # hooks every submodule, so the inner enc_c/gen params are counted). A plain
        # function fails with "'function' object has no attribute 'training'" and also
        # hides the inner params. So we wrap the pipeline in a composite module whose
        # submodules ARE enc_c and gen, then hand it to measure_flops (which
        # deep-copies + profiles an nn.Module correctly).
        if not meter._flops_done:
            class _CompositeGen(torch.nn.Module):
                def __init__(self, enc_c, gen, nz):
                    super(_CompositeGen, self).__init__()
                    self.enc_c = enc_c
                    self.gen = gen
                    self.nz = nz
                def forward(self, x):
                    zc = self.enc_c.forward_a(x)
                    zr = torch.randn(x.size(0), self.nz, device=x.device)
                    return self.gen.forward_b(zc, zr)
            enc_c_cpu = copy.deepcopy(model.enc_c).cpu().eval()
            gen_cpu = copy.deepcopy(model.gen).cpu().eval()
            composite = _CompositeGen(enc_c_cpu, gen_cpu, model.nz).cpu().eval()
            meter.measure_flops(composite, torch.randn(tuple(img1.shape)))
        for idx2 in range(opts.num):
            with torch.no_grad():
                img = model.test_forward(img1, a2b=opts.a2b)
            meter.time_inference(lambda: model.test_forward(img1, a2b=opts.a2b), idx1 * opts.num + idx2)
            imgs.append(img)
            names.append('output_{}_{}'.format(name[0].split('.')[0], idx2))
        save_imgs(imgs, names, result_dir)

    meter.report(result_dir)
    return


if __name__ == '__main__':
    main()

