# --*--coding   : uft-8  --*--
# @Time         : 2025/11/16 0:17
# @Author       : RisingInsight
# @File         : config.py
# @Software     : PyCharm
# @Project      : FSFNet-main
# @Function     :
import time

class TrainingConfig:
    def __init__(self):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

config = TrainingConfig()

