# quantizer/bitsandbytes.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import BaseQuantizer
import logging

class BitsAndBytesQuantizer(BaseQuantizer):
    def __init__(self, config: dict, model=None):
        super().__init__(config, model)
        self.model_name = config.get("model_name")
        self.quantization_type = config.get("quantization_type", "8bit")
        self.nf4 = config.get("nf4", False)
        self.double_quant = config.get("double_quant", False)
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)

    def _create_quantization_config(self):
        if self.quantization_type == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        else:
            quant_type = "nf4" if self.nf4 else "fp4"
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.double_quant,
            )

    def quantize(self):
        quant_config = self._create_quantization_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return self

    def save(self, path: str):
        if self.model is None or self.tokenizers is None:
            raise ValueError("Model not loaded.")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # 保存配置（可选）
        with open(os.path.join(path, "quant_config.txt"), "w") as f:
            f.write(f"quantization_type: {self.quantization_type}\n")
            if self.quantization_type == "4bit":
                f.write(f"nf4: {self.nf4}\n")
                f.write(f"double_quant: {self.double_quant}\n")

    def calibrate(self, data_loader):
        # bitsandbytes 量化不需要校准
        pass
