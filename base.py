# llm_quant/base.py
from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    """
    抽象基类：所有量化方法都要继承它
    """

    def __init__(self, model, **kwargs):
        """
        初始化量化器
        :param model: 传入的模型对象(transformers, torch.nn.Module 等）
        :param kwargs: 其他参数，比如量化位数
        """
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def quantize(self):
        """
        执行量化操作
        必须返回量化后的模型
        """
        pass

    @abstractmethod
    def save(self, save_dir: str):
        """
        保存量化后的模型
        """
        pass
