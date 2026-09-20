import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import cv2
from models.utils import make_coord
import torch


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A 'data/trainA'
    and from domain B 'data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot data'.
    Similarly, you need to prepare two directories:
    'data/testA' and 'data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path 'data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path 'data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from 'data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from 'data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.sr_H = opt.sr_H
        self.sr_W = opt.sr_W
        self.channels = opt.channels
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('L')
        A_img = np.array(A_img)
        B_img = np.array(B_img)
        # apply image transformation
        A = self.transform_A(Image.fromarray(A_img))
        B = self.transform_B(Image.fromarray(B_img))

        coord = make_coord([self.sr_H, self.sr_W, self.channels]) #coord:[HWC,3]
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / self.sr_H
        cell[:, 1] *= 2 / self.sr_W
        cell[:, 2] *= 2 / self.channels

        #return {'A': A, 'B': B, 'A_paths': A_path, 'coord_rgb':coord_rgb, 'cell_rgb':cell_rgb, 'coord_inf':coord_inf, 'cell_inf':cell_inf, 'B_paths': B_path}
        return {'A': A, 'B': B, 'A_paths': A_path, 'coord':coord, 'cell':cell, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


