class BaseQuantizer:
    def __init__(self, config: dict, model):
        self.config = config
        self.model = model

    def calibrate(self, data_loader):
        raise NotImplementedError

    def quantize(self):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError
