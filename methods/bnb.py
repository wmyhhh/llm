import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from ..base import BaseQuantizer
#from ..registry import register_quantizer

#@register_quantizer("bnb")
class BnBQuantizer(BaseQuantizer):
    """
    BitsAndBytes 量化实现
    支持 4-bit 和 8-bit 量化
    """

    def __init__(self, model, **kwargs):
        """
        :param model: HuggingFace 的模型名字 或 torch.nn.Module
        :param kwargs:
            - quant_type: "4bit" 或 "8bit" (默认 "4bit")
            - bnb_4bit_compute_dtype: torch.float16 / torch.bfloat16 等
            - device_map: 模型放置设备 (默认 "auto")
        """
        super().__init__(model, **kwargs)
        self.quantized = None

    def quantize(self):
        quant_type = self.kwargs.get("quant_type", "4bit")
        device_map = self.kwargs.get("device_map", "auto")
        bnb_4bit_compute_dtype = self.kwargs.get("bnb_4bit_compute_dtype", torch.float16)

        if isinstance(self.model, str):
            if quant_type == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
                )
                self.quantized = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    quantization_config=quant_config,
                    device_map=device_map,
                )
            elif quant_type == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self.quantized = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    quantization_config=quant_config,
                    device_map=device_map,
                )
            else:
                raise ValueError(f"Unsupported quant_type: {quant_type}")
        else:
            # 如果传入的是 nn.Module, 目前不做实际量化，直接返回
            print("[Warning] 手动传入 nn.Module 目前只返回原模型")
            self.quantized = self.model

        return self.quantized

    def save(self, save_dir: str):
        if self.quantized is None:
            raise RuntimeError("请先调用 quantize() 再保存模型")
        print(f"Saving quantized model to {save_dir} ...")
        self.quantized.save_pretrained(save_dir)
