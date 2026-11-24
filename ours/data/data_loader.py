"""
创建数据加载器
"""
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader         # 动态导入自定义数据集加载器类，避免循环导入问题
    data_loader = CustomDatasetDataLoader()                                     # 创建自定义数据集数据加载器实例
    print(data_loader.name())
    data_loader.initialize(opt)                                                 # 初始化数据加载器
    return data_loader
