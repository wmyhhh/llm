# quantizer/base.py

from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    def __init__(self, config: dict, model=None):
        """
        初始化基础量化器

        参数:
            config (dict): 量化配置参数
            model: 模型对象（可选）
        """
        self.config = config
        self.model = model

    @abstractmethod
    def calibrate(self, data_loader):
        """
        使用数据集进行校准（某些量化方法需要）
        """
        raise NotImplementedError("calibrate() 方法尚未实现")

    @abstractmethod
    def quantize(self):
        """
        执行量化操作
        """
        raise NotImplementedError("quantize() 方法尚未实现")

    @abstractmethod
    def save(self, path: str):
        """
        保存量化后的模型
        """
        raise NotImplementedError("save() 方法尚未实现")
