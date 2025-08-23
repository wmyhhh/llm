import os
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from ..registry import register_quantizer

@register_quantizer("gptq")
class GPTQQuantizer:
    def __init__(self, model, quant_type="4bit", device_map="auto", save_tokenizer=True, **kwargs):
        """
        GPTQ 量化器
        Args:
            model (str): HuggingFace 模型名称 或 本地路径
            quant_type (str): "4bit" 或 "8bit"
            device_map (str): "cuda" / "cpu" / "auto"
            save_tokenizer (bool): 是否保存 tokenizer
        """
        self.model_name_or_path = model
        self.quant_type = quant_type
        self.device_map = device_map
        self.save_tokenizer = save_tokenizer
        self.model = None
        self.quantized = False

    def quantize(self, calib_dataset=None, batch_size=1):
        """执行 GPTQ 量化"""
        bits = 4 if self.quant_type == "4bit" else 8
        quant_config = QuantizeConfig(bits=bits, group_size=128)

        print(f"[GPTQ] 加载模型 {self.model_name_or_path} ...")
        self.model = GPTQModel.load(self.model_name_or_path, quant_config)

        if calib_dataset is None:
            print("[GPTQ] 未提供校准数据，默认使用 C4 的 1024 条样本")
            calib_dataset = load_dataset(
                "allenai/c4",
                data_files="en/c4-train.00001-of-01024.json.gz",
                split="train"
            ).select(range(1024))["text"]

        print(f"[GPTQ] 开始量化 (bits={bits}) ...")
        self.model.quantize(calib_dataset, batch_size=batch_size)
        self.quantized = True
        return self.model

    def save(self, save_dir: str):
        """保存 GPTQ 模型"""
        if not self.quantized:
            raise RuntimeError("请先调用 quantize() 再保存！")

        os.makedirs(save_dir, exist_ok=True)
        print(f"[GPTQ] 保存量化模型到 {save_dir} ...")
        self.model.save(save_dir)

        if self.save_tokenizer:
            print("[GPTQ] 保存 tokenizer ...")
            self.model.tokenizer.save_pretrained(save_dir)
