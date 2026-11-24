"""
模型预训练
"""
import time
from options.pretrain_options import Options
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = Options().parse()                         # 解析命令行参数和配置
    data_loader = CreateDataLoader(opt)             # 创建数据加载器
    dataset = data_loader.load_data()               # 加载数据集
    dataset_size = len(data_loader)                 # 数据集图片数量
    print('#training images = %d' % dataset_size)
    model = create_model(opt)                       # 创建模型
    model.print_networks(verbose=True)              # 打印模型网络结构
    visualizer = Visualizer(opt)                    # 创建可视化工具，显示训练过程和结果
    total_steps = 0

    # 训练epoch轮
    for epoch in range(opt.epoch_count, opt.epochs + 1):
        epoch_start_time = time.time()              # 当前epoch开始时间
        epoch_iter = 0                              # 初始化当前epoch内步数计算器

        for i, data in enumerate(dataset):          # 遍历所有批次
            iter_start_time = time.time()           # 当前批次开始时间
            visualizer.reset()                      # 重置可视化器状态
            total_steps += opt.batchSize            # 按批次大小累加，更新总训练步数
            epoch_iter += opt.batchSize             # 更新当前epoch内步数
            model.set_input(data)                   # 将当前批次数据送入模型
            model.optimize_parameters()             # 执行模型的前向传播、损失计算、反向传播和参数优化

            # 显示当前训练结果
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 打印处理每个样本的平均时间，训练误差信息
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            # if total_steps

        # 保存模型，输出
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            # model.save('latest')
            model.save(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.epochs, time.time() - epoch_start_time))
        model.update_learning_rate()            # 更新学习率
