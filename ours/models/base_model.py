# --*--coding   : uft-8  --*--
# @Time         : 2026/2/16 12:15
# @Author       : RisingInsight
# @File         : base_model2.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import os
import torch
import time


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = self.opt.name_dir
        self.save_pretrain_dir = self.opt.name_dir
        self.load_dir = getattr(opt, 'load_dir''load_dir', self.save_dir)
        self.load_pretrain_dir = getattr(opt, 'pretrain_dir', '')

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            print(name)
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    # print(param)
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, gpu_ids):
        load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self.load_dir, load_filename)
        network.load_state_dict(torch.load(load_path))
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_pretrain_network(self, network, network_label, epoch_label, gpu_ids):
        load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self.load_pretrain_dir, load_filename)
        network.load_state_dict(torch.load(load_path), strict=False)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # update learning rate (called once every epoch)
