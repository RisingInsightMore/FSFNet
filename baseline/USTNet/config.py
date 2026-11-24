import time

class TrainingConfig:
    def __init__(self):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

config = TrainingConfig()

