### AECM
AECM: Attention-Enhanced Cross-Modality for Aerial Image and Video Translation with Vision Transformers

![hippo](https://github.com/Tracygc/AECM/blob/main/gif/AECM_demo_comparison0-60.gif)

![hippo](https://github.com/Tracygc/AECM/blob/main/gif/AECM_demo_comparison120-180.gif)

### Sample videos

Here's a sample videos:

# video1
real visible video1: https://github.com/user-attachments/assets/be0ebea7-d12d-4905-a0f1-bffc9f9e3026

fake infrared video1: https://github.com/user-attachments/assets/c3358c1a-415e-493c-8187-6544b9f82e75

real infrared video1: https://github.com/user-attachments/assets/5e50ec2d-8d03-48cf-8392-378aaa7dabda

# video2
real visible video2: https://github.com/user-attachments/assets/58ab55bc-103b-4ece-b77b-4f1b9a0887ff

fake infrared video2: https://github.com/user-attachments/assets/d439b667-4e60-4b07-b810-01d172dfb5d0

real infrared video2: https://github.com/user-attachments/assets/7ce08ee1-4f97-4d51-b378-d99dceb5b1e1


### Availability of Datasets
```
DroneVehicle dataset can be downloaded from <https://pan.baidu.com/s/1D4l3wXmAVSG2ywL6QLGURw?pwd=hrqf>, code: feqh. The updated AVIID dataset can be downloaded from https://pan.baidu.com/s/1M2WlHt1qqAuQ5QnUQSQ5lw?pwd=si8j Code: si8j.

InfraredCity and InfraredCity-Lite Dataset: The datasets and their more details are available in [InfiRay](http://openai.raytrontek.com/apply/Infrared_city.html/).
```
### Dataset Structure
```
dataset/
├── testA
├── testB
├── trainA
└── trainB
```

### Install Dependencies
```
Python 3.7 or higher
Pytorch 1.8.0, torchvison 0.9.0
Tensorboard, TensorboardX, Pyyaml, Pillow, dominate, visdom, timm, einops,  matplotlib, lpips, scikit-image, torch-fidelity
```

```
conda create translation python=3.13
conda activate translation
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Tensorboard (TensorboardX, Pyyaml, Pillow, dominate, visdom, einops,  matplotlib, lpips, scikit-image, torch-fidelity ) -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### Python
```
# Train for video mode
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /path --name AECM_name --dataset_mode unaligned_double --no_flip --local_nums 64 --display_env AECM_env --model aecm_double --side_length 7 --lambda_spatial 5.0 --lambda_global 5.0 --lambda_motion 1.0 --atten_layers 1,3,5 --lr 0.00001

# Train for image mode
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /path --name AECM_name --dataset_mode unaligned --local_nums 64 --display_env AECM_env --model aecm --side_length 7 --lambda_spatial 5.0 --lambda_global 5.0 --atten_layers 1,3,5 --lr 0.00001

##  Testing
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot /path/of/test_dataset --checkpoints_dir ./checkpoints --name train --model aecm --num_test 10000 --epoch latest
```

### Realism Evaluation
We use torch-fidelity (https://github.com/toshas/torch-fidelity) to evaluate the realism of the translated results.
```
FID
fidelity --gpu 0 --fid --input1  ./results/NightDrone_aecm/test_latest/images/fake_B --input2 ./results/NightDrone_aecm/test_latest/images/real_B
fidelity --gpu 0 --fid --input1  ./results/AVIID_aecm/test_latest/images/fake_B --input2 ./results/AVIID_aecm/test_latest/images/real_B 
KID
fidelity --gpu 0 --kid --input1  ./results/NightDrone_aecm/test_latest/images/fake_B --input2 ./results/NightDrone_aecm/test_latest/images/real_B  --kid-subset-size 1000
fidelity --gpu 0 --kid --input1  ./results/AVIID_aecm/test_latest/images/fake_B --input2 ./results/AVIID_aecm/test_latest/images/real_B  --kid-subset-size 150
```

### Reference Links
```
https://github.com/BIT-DA/ROMA
https://github.com/silver-hzh/DR-AVIT
https://github.com/silver-hzh/USTNet
```
