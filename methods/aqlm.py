# methods/aqlm.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..base import BaseQuantizer
#from . import register_quantizer

#@register_quantizer("aqlm")
class AQLMQuantizer(BaseQuantizer):
    def __init__(self, model, device_map="auto", save_tokenizer=False, quant_type=None, **kwargs):
        super().__init__(model, device_map, save_tokenizer, quant_type, **kwargs)
        self.tokenizer = None

    def quantize(self):
        """
        对于 AQLM 来说，我们加载的就是已经预量化的模型，
        所以这里主要做加载和初始化，而不是传统意义的在线量化。
        """
        # 检查 aqlm 是否安装
        try:
            import aqlm
        except ImportError:
            raise ImportError("请先安装 aqlm: pip install aqlm[gpu,cpu]")

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[AQLMQuantizer] 加载预量化模型: {self.model_name_or_path}")

        # 加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="auto" if "cuda" in self.device else None,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[AQLMQuantizer] 加载失败: {e}")
            raise e

        print("[AQLMQuantizer] 模型加载成功")
        return self.model, self.tokenizer

