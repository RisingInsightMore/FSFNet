import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)         # 创建数据加载器，加载非配对数据集
    dataset = data_loader.load_data()           # 从数据加载器中加载数据集
    dataset_size = len(data_loader)             # 获取数据集大小（批次数量）
    print('#training images = %d' % dataset_size)
    model = create_model(opt)                   # 创建模型，根据选项配置选择模型类型，初始化模型
    model.print_networks(verbose=False)         # 打印模型网络结构，verbose=False表示不打印详细信息
    visualizer = Visualizer(opt)                # 创建可视化器，用于可视化训练过程中的结果
    total_steps = 0

    # 训练循环，遍历每个epoch，每个epoch遍历整个数据集，每个批次进行一次前向传播和反向传播
    for epoch in range(opt.epoch_count, opt.epochs_warmup + opt.epochs_anneal + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)           # self.input_A, self.input_B的形状为：(nb, channels, H, W)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            # if total_steps
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            # model.save('latest')
            model.save(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.epochs_warmup + opt.epochs_anneal, time.time() - epoch_start_time))
        model.update_learning_rate()
