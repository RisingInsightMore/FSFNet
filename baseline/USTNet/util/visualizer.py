import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from config import config



class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id                    # 显示ID，控制是否使用visdom
        self.use_html = opt.isTrain and not opt.no_html     # 是否使用HTML保存结果
        self.win_size = opt.display_winsize                 # 显示窗口大小
        self.name = opt.name                                # 实验名称
        self.opt = opt
        self.saved = False
        # 为每个实验创建唯一标识
        if hasattr(self.opt, 'experiment_id'):
            self.experiment_id = self.opt.experiment_id
        else:
            self.experiment_id = f"{self.opt.name}_{config.timestamp}"

        # 标记是否已保存
        if self.display_id > 0:         # 不展示，display_id设置为0了。否则初始化visdom连接
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:               # 创建html展示文件路径
            self.web_dir = os.path.join(opt.checkpoints_dir, self.experiment_id, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, self.experiment_id, 'loss_log.txt')     # 创建损失日志文件
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):            # 重置保存状态
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        """
                显示或保存当前结果
                参数：
                    visuals：包含要显示或保存图像的字典
                    epoch：当前epoch数
                    save_result：是否保存结果
        """
        if self.display_id > 0:  # show images in the browser，使用visdom在浏览器中显示图像
            ncols = self.opt.display_single_pane_ncols          # 每行显示的图像数量
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]   # 获取图像尺寸
                # 定义CSS样式
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))           # 计算行数
                images = []
                idx = 0

                # 处理每个图像
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label                 # 添加标签
                    images.append(image_numpy.transpose([2, 0, 1]))         # 转换维度为CxHxW
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''

                # 用白色图像填充空单元格
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                # 在visdom中显示图像和标签
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                # 单独窗口显示每个图像
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file，保存图像到HTML文件
            self.saved = True
            # 保存当前epoch的图像
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)    # update website, 更新网页
            # 从当前epoch反向遍历到1
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        """
                绘制当前错误/损失
                参数:
                    epoch: 当前epoch
                    counter_ratio: 当前epoch内的迭代比例
                    opt: 选项
                    errors: 错误标签和值的字典
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}          # 初始化绘图数据
        self.plot_data['X'].append(epoch + counter_ratio)                               # 添加新数据点
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        # 在visdom中绘制损失曲线
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        """
                打印当前错误/损失到控制台和日志文件
                参数:
                    epoch: 当前epoch
                    i: 当前迭代
                    errors: 错误字典
                    t: 时间
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:      # 写入日志文件
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        """
                保存图像到磁盘

                参数:
                    webpage: HTML网页对象
                    visuals: 可视化图像字典
                    image_path: 原始图像路径
        """
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])         # 获取基础文件名
        name = os.path.splitext(short_path)[0]              # 去除扩展名

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_images_iter(self, webpage, visuals, image_path, iter_num):
        """
                按迭代次数保存图像到磁盘
                参数:
                    webpage: HTML网页对象
                    visuals: 可视化图像字典
                    image_path: 原始图像路径
                    iter_num: 迭代次数
        """
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = str(iter_num)                            # 使用迭代次数作为名称

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

