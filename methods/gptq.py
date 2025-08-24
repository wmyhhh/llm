import os
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from ..base import BaseQuantizer

class GPTQQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        device_map="auto",
        quant_type="4bit",  # 支持 "2bit", "3bit", "4bit"
        save_tokenizer=True,
        save_dir=None,
        **kwargs
    ):
        """
        GPTQ 量化器，支持 2/3/4 bit 量化
        :param model: HuggingFace 模型名 或 torch.nn.Module
        :param kwargs: 其他参数
        """
        super().__init__(
            model=model,
            device_map=device_map,
            quant_type=quant_type,
            save_tokenizer=save_tokenizer,
            save_dir=save_dir or "gptq",
            **kwargs
        )

    def quantize(self, calib_dataset=None, batch_size=1):
        """执行 GPTQ 量化"""
        allowed_bits = [2, 3, 4]
        # 从 quant_type 字符串提取位数
        bits = int(self.quant_type.replace("bit", ""))
        if bits not in allowed_bits:
            raise ValueError(f"[GPTQ] Unsupported bits: {bits}. Allowed: {allowed_bits}")

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

    def save(self, save_dir: str = None):
        """保存 GPTQ 模型"""
        save_dir = save_dir or self.save_dir
        if not self.quantized:
            raise RuntimeError("[GPTQ] 请先调用 quantize() 再保存！")

        os.makedirs(save_dir, exist_ok=True)
        print(f"[GPTQ] 保存量化模型到 {save_dir} ...")
        self.model.save(save_dir)

        if self.save_tokenizer and hasattr(self.model, "tokenizer"):
            print("[GPTQ] 保存 tokenizer ...")
            self.model.tokenizer.save_pretrained(save_dir)
