# --*--coding   : uft-8  --*--
# @Time         : 2026/2/22 16:18
# @Author       : RisingInsight
# @File         : test2.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

if __name__ == '__main__':
    opt = TestOptions().parse()

    # 测试时强制覆盖某些参数以保证确定性
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True      # 按顺序取数据，不打乱
    opt.no_flip = True              # 禁止随机翻转
    # 根据 --test_preprocess 设置 resize_or_crop
    opt.resize_or_crop = 'resize_and_crop'
    # 如果选择 'none'，可能需要调整 loadSize 和 fineSize 的关系，这里简单处理
    if opt.test_preprocess == 'none':
        opt.loadSize = opt.fineSize   # 确保图像不缩放，直接使用原始尺寸（可能不符合网络输入）

    # 数据加载（使用 phase='test'，需确保数据集中有 testA 等）
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f'Loaded {len(dataset)} test images.')

    # 创建模型（此时会调用 FSFNet.initialize，并根据 isTrain=False 自动加载指定 epoch 的权重）
    model = create_model(opt)
    model.print_networks(verbose=False)

    # 验证模型的 save_dir 是否正确（应指向 opt.name_dir）
    print(f"Model save directory: {model.save_dir}")

    visualizer = Visualizer(opt)

    # 构建输出目录：results_dir/name/phase_epoch/
    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.which_epoch}")
    os.makedirs(web_dir, exist_ok=True)
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.which_epoch}')

    # 测试循环
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print(f'{i:04d}: process image... {img_path}')
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
    print(f'Results saved to {web_dir}')




