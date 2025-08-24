# methods/awq.py
from ..base import BaseQuantizer
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

class AWQQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        device_map="auto",
        quant_type="4bit",  # AWQ 默认支持 2/3/4bit
        save_tokenizer=True,
        save_dir=None,
        **kwargs
    ):
        """
        :param model: HuggingFace 模型名或本地路径
        :param quant_type: 量化位数，支持 2, 3, 4
        :param kwargs: 其他参数，例如 zero_point, q_group_size, version
        """
        super().__init__(
            model=model,
            device_map=device_map,
            quant_type=quant_type,
            save_tokenizer=save_tokenizer,
            save_dir=save_dir or "awq",
            **kwargs
        )
        self.tokenizer = None

    def quantize(self):
        # 确保位数有效
        allowed_bits = [2, 3, 4]
        bits = int(self.quant_type.replace("bit", ""))
        if bits not in allowed_bits:
            raise ValueError(f"AWQ only supports {allowed_bits} bits, got {bits}")

        print(f"[AWQ] Loading model {self.model_name_or_path}")
        self.model = AutoAWQForCausalLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        quant_config = {
            "w_bit": bits,
            "zero_point": self.kwargs.get("zero_point", True),
            "q_group_size": self.kwargs.get("q_group_size", 128),
            "version": self.kwargs.get("version", "GEMM")
        }

        print(f"[AWQ] Quantizing with config: {quant_config}")
        self.model.quantize(self.tokenizer, quant_config=quant_config)
        self.quantized = True
        return self.model

    def save(self, path=None):
        save_path = path or self.save_dir
        os.makedirs(save_path, exist_ok=True)
        if not self.quantized:
            raise RuntimeError("请先调用 quantize() 再保存模型")
        print(f"[AWQ] Saving quantized model to {save_path}")
        self.model.save_quantized(save_path)
        if self.save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
        return save_path
