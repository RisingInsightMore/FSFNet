# evaluate_dravit.py
# DRAVIT baseline 的统一指标评估入口（可见光 -> 红外 翻译）。
#
# ---------------------------------------------------------------------------
# 为什么需要单独写这个脚本（与 AECM 的差异）
# ---------------------------------------------------------------------------
# 本仓库所有 baseline 的像素级指标（LPIPS / SSIM / PSNR / RMSE）以及
# --fid 时的 FID / KID 口径，都由统一引擎 baseline/scripts/eval_core.py 的
# run_evaluation() 保证，与 AECM 逐字一致（见 eval_core.py 头部说明）。
# 本脚本不重写任何指标计算逻辑，只负责把 DRAVIT 特殊结构的产物整理成
# run_evaluation 能理解的标准配对，再委托它完成全部打分。
#
# DRAVIT 的输出结构与 AECM 不同，必须自定义配对：
#   DRAVIT 的测试入口 baseline/DRAVIT/DR-AVIT/src/test.py 将结果写到
#   ../outputs/<name>/，再由 run_test.sh 整体拷到统一目录
#   ./results/DRAVIT_AVIID/test_<epoch>/images/，其中 <epoch> 是所用权重的
#   5 位 epoch 编号（例如用 00399.pth 推理即 test_00399，与训练轮数无关）。
#   该目录是【扁平结构】：
#     - GT（红外真值）：文件名 <stem>.<ext>（如 001.png），来自 testB，
#       无前缀无后缀。
#     - 生成图：output_<stem>_<idx>.<ext>（如 output_001_0.png …
#       output_001_9.png），来自 testA 经模型翻译。DRAVIT 是多模态翻译，
#       --num 默认 10，即每个输入默认生成 10 张（idx 0..9）。
#   即：真实图文件名不带 output_ 前缀；生成图文件名以 output_ 开头、末尾
#   带 _<数字> 表示模态序号。
#
#   eval_core.find_pairs 假设的是 fake_B/real_B 子目录或 *_fake_B/*_real_B
#   扁平命名，无法处理 DRAVIT 这种命名，会把 GT 漏掉导致 0 配对。因此本脚本
#   自行扫描并按 --mode 选出一个模态的生成图，staging 成两个临时目录后交给
#   run_evaluation 的 --real/--fake 分支（其 _key 配对在"同名"时自然成立）。
#
# ---------------------------------------------------------------------------
# 用法示例（与统一约定一致）:
#   cd ./baseline/DRAVIT && python evaluate_dravit.py \
#       --input ./results/DRAVIT_AVIID/test_<epoch>/images \
#       --out ./eval_DRAVIT_AVIID --fid
# 可选 --mode 指定取第几个模态（默认 0）作为评估用生成图。
#
# 验收要点：stage_pairs(input_dir, mode) 是纯文件操作（不依赖 torch/lpips），
# 可在无 GPU 环境下做单元测试：mock 出 input 目录即可。
import argparse
import os
import re
import shutil
import sys
import tempfile
import warnings

# 把统一引擎所在目录加入 sys.path，复用 run_evaluation（与 AECM 同一口径）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_SCRIPT_DIR, '..', 'scripts')
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

from eval_core import run_evaluation  # noqa: E402  (依赖上方 sys.path 注入)


# ---------------------------------------------------------------------------
# 配对与 staging（纯 Python / 纯文件操作，不 import torch / lpips）
# ---------------------------------------------------------------------------
# 生成图命名：output_<stem>_<idx>.<ext>
#   group(1) -> stem（与真实图文件名 stem 对应）；group(2) -> 模态 idx（int）。
_OUTPUT_RE = re.compile(r'^output_(.+)_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
# 真实图：图片文件且不以 output_ 开头（GT 来自 testB，无前缀）。
_IMG_SUFFIXES = ('.png', '.jpg', '.jpeg')


def stage_pairs(input_dir, mode=0):
    """扫描 DRAVIT 扁平输出目录并按 --mode 选出一个模态的生成图，
    把配对的两组分发到两个临时目录（real_dir / fake_dir），fake 落盘文件名
    用 <stem>.<ext>（与 real 同名，便于 run_evaluation 的 _key 配对）。

    参数:
        input_dir (str): ./results/DRAVIT_*/test_<epoch>/images 目录
                        （<epoch> 为所用权重的 5 位编号，如 00399）。
        mode (int):      选用的模态序号，默认 0。

    返回:
        (real_dir, fake_dir, n_pairs, skipped)
        real_dir  -> 真实图临时目录，文件名 <stem>.<ext>
        fake_dir  -> 生成图临时目录，文件名 <stem>.<ext>（与 real 同名）
        n_pairs   -> 成功配对的 (real, fake) 数量
        skipped   -> 缺失 mode 模态的真实图 stem 数量

    注意:
        - 该函数只做文件拷贝，不计算任何指标。
        - 返回的临时目录由调用方在 run_evaluation 结束后负责清理。
        - 缺失目标模态的 stem 会跳过并通过 warnings 输出日志。
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError('input 目录不存在: %s' % input_dir)

    real_candidates = {}  # stem -> (src_path, ext)
    fake_index = {}       # stem -> {idx: (src_path, ext)}

    for fname in os.listdir(input_dir):
        lower = fname.lower()
        if not lower.endswith(_IMG_SUFFIXES):
            continue
        ext = os.path.splitext(fname)[1]
        m = _OUTPUT_RE.match(fname)
        if m:
            # 生成图：output_<stem>_<idx>.<ext>
            stem = m.group(1)
            idx = int(m.group(2))
            fake_index.setdefault(stem, {})[idx] = (os.path.join(input_dir, fname), ext)
        elif not fname.startswith('output_'):
            # 真实图（GT）：不以 output_ 开头、且为图片
            stem = os.path.splitext(fname)[0]
            real_candidates[stem] = (os.path.join(input_dir, fname), ext)

    real_dir = tempfile.mkdtemp(prefix='dravit_real_')
    fake_dir = tempfile.mkdtemp(prefix='dravit_fake_')

    n_pairs = 0
    skipped = 0
    # 按 stem 排序，保证结果可复现。
    for stem in sorted(real_candidates):
        real_src, ext = real_candidates[stem]
        modalities = fake_index.get(stem)
        if not modalities or mode not in modalities:
            skipped += 1
            warnings.warn(
                'stem=%s 缺失模态 mode=%d，已跳过（可用模态: %s）'
                % (stem, mode, sorted(modalities) if modalities else []),
                UserWarning,
                stacklevel=2,
            )
            continue
        fake_src, _ = modalities[mode]
        # 落盘文件名用 <stem>.<ext>，real 与 fake 同名 -> _key 配对成立。
        shutil.copy2(real_src, os.path.join(real_dir, stem + ext))
        shutil.copy2(fake_src, os.path.join(fake_dir, stem + ext))
        n_pairs += 1

    return real_dir, fake_dir, n_pairs, skipped


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DRAVIT visible-to-infrared translation results.')
    parser.add_argument('--input', type=str, required=True,
                        help='DRAVIT results images dir, '
                             'e.g. ./results/DRAVIT_AVIID/test_400/images')
    parser.add_argument('--out', type=str, required=True,
                        help='dir to save results.txt')
    parser.add_argument('--fid', action='store_true',
                        help='also compute FID/KID (needs torch_fidelity installed)')
    parser.add_argument('--mode', type=int, default=0,
                        help='which modality (0-based) to use as the generated '
                             'image; DRAVIT outputs --num (default 10) per input')
    opt = parser.parse_args()

    # ---- 开头说明：DRAVIT 多模态特性 + 当前评估模态 ----
    print('=' * 64)
    print('DRAVIT Evaluation')
    print('  DRAVIT 为多模态翻译，默认每个输入生成 --num(=10) 张图'
          '（output_<stem>_0..9.png）。')
    print('  本脚本固定取第 %d 个模态（0-based）作为评估用生成图。'
          % opt.mode)
    print('  input : %s' % opt.input)
    print('  out   : %s' % opt.out)
    print('  fid   : %s' % ('ON' if opt.fid else 'OFF'))
    print('=' * 64)

    # ---- staging：扫描并按 --mode 选 fake，落到临时目录 ----
    real_dir, fake_dir, n_pairs, skipped = stage_pairs(opt.input, opt.mode)

    print('[stage] real dir=%s' % real_dir)
    print('[stage] fake dir=%s' % fake_dir)
    print('[stage] paired=%d  skipped(缺模态)=%d' % (n_pairs, skipped))
    if n_pairs == 0:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)
        sys.exit('ERROR: 在 %s 中按 mode=%d 未找到任何配对，'
                 '请确认目录为 DRAVIT 扁平输出结构。' % (opt.input, opt.mode))

    # ---- 委托统一引擎打分（与 AECM 逐字一致）----
    try:
        run_evaluation(
            'DRAVIT',
            argv=['--real', real_dir, '--fake', fake_dir,
                  '--out', opt.out] + (['--fid'] if opt.fid else []),
        )
    finally:
        # ---- 清理临时 staging 目录 ----
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
