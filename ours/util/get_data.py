from __future__ import print_function
import os
import tarfile
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup
from os.path import abspath, isdir, join, basename


class GetData(object):
    """

    Download CycleGAN or Pix2Pix Data.

    Args:
        technique : str
            One of: 'cyclegan' or 'pix2pix'.
        verbose : bool
            If True, print additional information.

    Examples:
        >>> from util.get_data import GetData
        >>> gd = GetData(technique='cyclegan')
        >>> new_data_path = gd.get(save_path='./datasets')  # options will be displayed.

    """

    def __init__(self, technique='cyclegan', verbose=True):
        url_dict = {
            'pix2pix': 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets',
            'cyclegan': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets'
        }
        self.url = url_dict.get(technique.lower())          # 获取指定数据集URL
        self._verbose = verbose                             # 是否打印详细信息

    def _print(self, text):
        if self._verbose:
            print(text)

    # 获取数据集选项
    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text, 'lxml')
        options = [h.text for h in soup.find_all('a', href=True)
                   if h.text.endswith(('.zip', 'tar.gz'))]
        return options

    # 展示数据集选项
    def _present_options(self):
        r = requests.get(self.url)              # 获取数据集URL对应的HTML页面
        options = self._get_options(r)          # 从HTML页面中提取数据集选项
        print('Options:\n')
        for i, o in enumerate(options):
            print("{0}: {1}".format(i, o))
        choice = input("\nPlease enter the number of the "
                       "dataset above you wish to download:")
        return options[int(choice)]

    # 下载数据集
    def _download_data(self, dataset_url, save_path):
        if not isdir(save_path):
            os.makedirs(save_path)

        base = basename(dataset_url)
        temp_save_path = join(save_path, base)

        with open(temp_save_path, "wb") as f:           # 以二进制写入模式打开临时文件，用于保存下载的数据
            r = requests.get(dataset_url)               # 发送GET请求，获取数据集URL对应的响应
            f.write(r.content)                          # 将响应内容写入临时文件

        if base.endswith('.tar.gz'):                    # tar.gz格式
            obj = tarfile.open(temp_save_path)
        elif base.endswith('.zip'):
            obj = ZipFile(temp_save_path, 'r')
        else:
            raise ValueError("Unknown File Type: {0}.".format(base))

        self._print("Unpacking Data...")
        obj.extractall(save_path)
        obj.close()
        os.remove(temp_save_path)

    def get(self, save_path, dataset=None):
        """

        Download a dataset.

        Args:
            save_path : str
                A directory to save the data to.
            dataset : str, optional
                A specific dataset to download.
                Note: this must include the file extension.
                If None, options will be presented for you
                to choose from.

        Returns:
            save_path_full : str
                The absolute path to the downloaded data.

        """
        if dataset is None:
            selected_dataset = self._present_options()
        else:
            selected_dataset = dataset

        save_path_full = join(save_path, selected_dataset.split('.')[0])

        if isdir(save_path_full):
            warn("\n'{0}' already exists. Voiding Download.".format(
                save_path_full))
        else:
            self._print('Downloading Data...')
            url = "{0}/{1}".format(self.url, selected_dataset)
            self._download_data(url, save_path=save_path)

        return abspath(save_path_full)