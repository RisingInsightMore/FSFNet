# -*- coding: utf-8 -*-
#
# count_params.py —— DRAVIT 关键子模块参数量统计脚本
# =============================================================
# 用途：
#     为 USTNet 项目的 DRAVIT 基线模型，在论文的
#     “# of Parameters in Key Modules”（各关键子模块参数数量）
#     实验中，产出可直接抄写进论文表格的数据。
#
# 重要特性（无需 checkpoint、无需 GPU）：
#     参数数量只由“网络结构”决定，与训练权重无关，因此本脚本：
#       * 不加载任何 checkpoint（不调用 model.resume）；
#       * 不调用 setgpu / 不碰 CUDA（纯 CPU 即可运行）；
#       * 使用 TrainOptions().parse() 直接构建 opts，无需任何
#         外部输入（无数据集、无权重、无 dataloader）。
#     本脚本可在纯 CPU、无数据、无权重的环境下稳定产出参数量。
#
# 与训练/推理日志的一致性（关键）：
#     本脚本的“标准逐行输出”部分直接复用 model.print_networks(False)，
#     因此与 train.py / test.py 在训练/推理时打印的内容逐行完全一致
#     （均为 `[Network X] Total number of parameters : %.3f M` 格式）。
#
# 两种“生成器”口径说明（仅用于下方聚合块，不混入上面的标准日志）：
#     * translation_generator = enc_c + gen
#           即 a2b 推理真正使用的网络（见 model.test_forward 仅用 enc_c+gen），
#           实测 ≈ 26.16M（本脚本统计值）。注：早期 efficiency.py 的 13.67M 与当前
#           结构不符（gen 现 17.146M、enc_c 现 9.018M），以本脚本为准。
#     * generator_stack = enc_c + enc_a + gen
#           用户选定的“完整生成网络”口径，论文 "Parameters in Key Modules" 表中
#           DRAVIT 的“生成器”行使用此值；enc_a 仅用于风格迁移/训练，不参与基础 A→B 翻译。
#
# 运行方式：
#     python count_params.py
#
# 输出：
#     1) 标准逐行输出（与 train.py / test.py 完全一致）+ 清晰的论文聚合块；
#     2) 二者一起写入 result_dir/params_breakdown.txt（result_dir 取 opts 默认值）。
# =============================================================

import contextlib
import io
import os
import sys

# 将 DR-AVIT/src 目录加入 sys.path，使 `from options import TrainOptions`
# 与 `from model import DR_AVIT` 能正确解析到模型源码所在目录。
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DR-AVIT', 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402,F401
from options import TrainOptions  # noqa: E402
from model import DR_AVIT  # noqa: E402


def count_params(module) -> int:
    """统计一个 nn.Module 的全部可学习参数数量（含所有子模块，求和）。

    Args:
        module (torch.nn.Module | None): 需要统计的 PyTorch 模块；
            若为 None，返回 0。

    Returns:
        int: 参数张量中元素的总数（所有 p.numel() 之和）。
    """
    if module is None:
        return 0
    total = 0
    for p in module.parameters():
        total += int(p.numel())
    return total


def _to_m(n: int) -> float:
    """将参数整数数量转换为百万（M）单位。"""
    return n / 1_000_000.0


# DRAVIT 中 10 个判别器的属性名清单（顺序：域判别器 -> 几何一致性判别器）。
_DISCRIMINATOR_NAMES = (
    'disA', 'disA_gc', 'disB', 'disB_gc',
    'disA2', 'disA2_gc', 'disB2', 'disB2_gc',
    'disContent', 'disContent_gc',
)


def main() -> None:
    """统计并打印 DRAVIT 关键子模块参数数量，并落盘到 result_dir。"""
    # 1) 构建 opts：CPU、无外部依赖、不加载数据。
    opts = TrainOptions().parse()

    # 2) 实例化模型：仅构建网络结构（CPU），不加载权重、不碰 CUDA。
    model = DR_AVIT(opts)

    # 3) 标准逐行输出：直接复用 model.print_networks(False)，
    #    与 train.py / test.py 在训练/推理时打印的内容逐行完全一致。
    #    通过重定向 stdout 捕获，便于随后与聚合块一起落盘。
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.print_networks(False)
    standard_block = buf.getvalue()
    # 原样打印到控制台（含头尾分隔线，与 baseline 一致）。
    print(standard_block, end='')

    # 4) 论文填表聚合块（清晰分隔，不混入上面的训练/推理日志）。
    enc_c = count_params(model.enc_c)              # 内容/语义编码器
    enc_a = count_params(model.enc_a)              # 风格编码器
    gen = count_params(model.gen)                  # 跨域生成器
    # translation_generator = enc_c + gen：a2b 推理真正使用的网络（见 test_forward），
    # 实测 ≈ 26.16M（本脚本统计值；早期 efficiency.py 的 13.67M 已不符当前结构）。
    translation_generator = enc_c + gen
    # generator_stack = enc_c + enc_a + gen：用户选定的“完整生成网络”口径
    # （论文 "Parameters in Key Modules" 表中 DRAVIT 的“生成器”行使用此值）。
    generator_stack = enc_c + enc_a + gen

    discriminator_total = 0
    for name in _DISCRIMINATOR_NAMES:
        discriminator_total += count_params(getattr(model, name))

    model_total = count_params(model)             # 全模型（所有参数）

    agg_lines = [
        '',
        '===== DRAVIT paper-table aggregates (NOT part of train/test log) =====',
        '    %-26s %12s %12s   %s' % ('Aggregate', 'Params', 'Params(M)', 'Note'),
        '    ' + '-' * 78,
        '    %-26s %12d %10.4f M   %s' % (
            'translation_generator', translation_generator, _to_m(translation_generator),
            '(= enc_c + gen; a2b inference net; measured 26.16M, NOT 13.67M)'),
        '    %-26s %12d %10.4f M   %s' % (
            'generator_stack', generator_stack, _to_m(generator_stack),
            '(= enc_c + enc_a + gen; full generation network, user-selected)'),
        '    %-26s %12d %10.4f M   %s' % (
            'dis_total', discriminator_total, _to_m(discriminator_total),
            '(sum of 10 discriminators)'),
        '    %-26s %12d %10.4f M   %s' % (
            'model_total', model_total, _to_m(model_total),
            '(sum of all 13 sub-networks)'),
        '===== end of paper-table aggregates =====',
        '',
    ]
    agg_block = '\n'.join(agg_lines)
    print(agg_block)

    # 5) 落盘：标准逐行输出 + 聚合块 一起写入 result_dir/params_breakdown.txt。
    full_report = standard_block + agg_block
    try:
        result_dir = getattr(opts, 'result_dir', '../results')
        os.makedirs(result_dir, exist_ok=True)
        out_path = os.path.join(result_dir, 'params_breakdown.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print('[INFO] 参数明细已保存至: %s' % os.path.abspath(out_path))
    except Exception as e:  # noqa: BLE001
        print('[WARN] 参数明细落盘失败（不影响 stdout 输出）：%s' % e)


if __name__ == '__main__':
    main()
