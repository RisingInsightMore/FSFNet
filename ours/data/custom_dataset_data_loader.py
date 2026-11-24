import torch.utils.data
from data.base_data_loader import BaseDataLoader


# 数据集工厂函数：根据配置创建对应的数据集实例
def CreateDataset(opt):
    dataset = None                          # 初始化数据集变量
    if opt.dataset_mode == 'unaligned':     # 根据数据集模式选择创建对应的数据集
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()        # 创建非配对数据集实例

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)                 # 使用配置选项初始化数据集
    return dataset

# 自定义数据集数据加载器类，继承自基础数据加载器
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    # 初始化数据加载器
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)               # 使用工厂函数创建数据集实例
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,             # serial_batches为False，随机打扰数据集
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:          # 若达到最大数据集大小限制，停止迭代
                break
            yield data                                  # 使用yield返回当前批次数据，保持迭代状态
