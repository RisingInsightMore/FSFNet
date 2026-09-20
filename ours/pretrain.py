# --*--coding   : uft-8  --*--
# @Time         : 2026/2/16 10:35
# @Author       : RisingInsight
# @File         : pretrain2.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
"""
模型预训练（支持断点续训）
"""
import time
from options.pretrain_options import Options
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = Options().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.print_networks(verbose=True)
    visualizer = Visualizer(opt)
    total_steps = 0

    # 确定起始 epoch（支持断点续训）
    if hasattr(model, 'epoch'):
        start_epoch = model.epoch + 1
    else:
        start_epoch = opt.epoch_count

    # 训练循环
    for epoch in range(start_epoch, opt.epochs + 1):
        model.epoch = epoch          # 更新模型中的 epoch 属性，便于保存时使用
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            # 显示训练结果
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 打印训练误差
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        # 按 epoch 频率保存模型
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)                # 保存网络权重（用于后续微调）
        model.save_checkpoint('latest')     # 每次epoch都更新latest

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.epochs, time.time() - epoch_start_time))
        model.update_learning_rate()