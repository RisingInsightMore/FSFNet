# ImconfuseNet Baseline

Implicit Continuous Full-spectrum Response Network for Visible to Infrared Image Translation (TGRS, 2026)

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- scipy
- pillow
- opencv-python
- dominate
- requests
- beautifulsoup4
- tensorboardX

## 数据集

支持三个数据集：
- AVIID
- DayDrone
- NightDrone

数据集通过符号链接存放在 `datasets/` 目录下。

## 使用方法

### 验证安装

```bash
bash verify_installation.sh
```

### 训练

```bash
# 使用 AVIID 数据集
bash run_train.sh --dataset AVIID --gpu_ids 0

# 使用 DayDrone 数据集
bash run_train.sh --dataset DayDrone --gpu_ids 0

# 使用 NightDrone 数据集
bash run_train.sh --dataset NightDrone --gpu_ids 0

# 自定义参数
bash run_train.sh --dataroot ./datasets/AVIID --name MyExperiment --epochs 200 --gpu_ids 0
```

### 测试

```bash
# 使用 AVIID 数据集
bash run_test.sh --dataset AVIID --epoch latest --gpu_ids 0

# 使用 DayDrone 数据集
bash run_test.sh --dataset DayDrone --epoch latest --gpu_ids 0

# 使用 NightDrone 数据集
bash run_test.sh --dataset NightDrone --epoch latest --gpu_ids 0

# 自定义参数
bash run_test.sh --dataroot ./datasets/AVIID --name MyExperiment --epoch 400 --gpu_ids 0
```

## 结果保存

- 训练权重：`./checkpoints/{NAME}/`
- 测试结果：`../results/{NAME}/test_{EPOCH}/images`

## 原论文参数

- batch_size: 8
- load_size: 128
- crop_size: 128
- image_size: 128
- sr_H: 128
- sr_W: 128
- epochs: 400

## 引用

```bibtex
@article{ImconfuseNet2026,
  title={Implicit Continuous Full-spectrum Response Network for Visible to Infrared Image Translation},
  author={},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```
