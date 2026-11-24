import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser('split USTNet image')
parser.add_argument('--input', help='input directory for src image')
args = parser.parse_args()


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def copy_file(src_path, target_path):
    shutil.move(src=src_path, dst=target_path)


def remove_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)


fake_path = os.path.join(args.input + '_fake')
if os.path.exists(fake_path):
    remove_file(fake_path)
else:
    makedir(fake_path)
real_path = os.path.join(args.input + '_real')
if os.path.exists(real_path):
    remove_file(real_path)
else:
    makedir(real_path)


# 要要权限，才能复制文件
# input_img_path = os.path.join(args.input + '_copy')
# copy_file(src_path=os.path.join(args.input), target_path=os.path.join(input_img_path))

input_img_path = os.path.join(args.input)                       # 输入路径

filename_list = [x for x in os.listdir(input_img_path)]         # 输入路径下的所有文件名
sample_list = []
for name in filename_list:
    if 'real_B' in name:
        copy_file(src_path=os.path.join(input_img_path, name), target_path=os.path.join(real_path, name))
    elif 'fake_B' in name:
        copy_file(src_path=os.path.join(input_img_path, name), target_path=os.path.join(fake_path, name))

