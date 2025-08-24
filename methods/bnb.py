import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from ..base import BaseQuantizer
#from ..registry import register_quantizer

#@register_quantizer("bnb")
class BnBQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        device_map="auto",
        quant_type="4bit",  # bnb 一般是 4bit/8bit
        save_tokenizer=True,
        save_dir=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            device_map=device_map,
            quant_type=quant_type,
            save_tokenizer=save_tokenizer,
            save_dir=save_dir or "bnb",
            **kwargs
        )
        
    def quantize(self):
        quant_type = self.kwargs.get("quant_type", "4bit")
        device_map = self.kwargs.get("device_map", "auto")
        bnb_4bit_compute_dtype = self.kwargs.get("bnb_4bit_compute_dtype", torch.float16)

        if isinstance(self.model_name_or_path, str):
            if quant_type == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
                )
                self.quantized = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config=quant_config,
                    device_map=device_map,
                )
            elif quant_type == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self.quantized = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
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
