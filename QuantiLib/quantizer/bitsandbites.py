#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 bitsandbytes 量化模型的简化实现
支持 8-bit 和 4-bit 量化
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

class BABQuant:
    """
    使用 BitsAndBytes 进行模型量化的类
    """
    def __init__(self, model_name, quantization_type="8bit", nf4=False, double_quant=False):
        """
        初始化量化器
        
        参数:
            model_name (str): 要量化的模型名称或路径
            quantization_type (str): 量化类型，可选 "8bit" 或 "4bit"
            nf4 (bool): 是否使用 NF4 数据类型进行 4 位量化
            double_quant (bool): 是否使用嵌套量化进一步减少内存使用
        """
        self.model_name = model_name
        self.quantization_type = quantization_type
        self.nf4 = nf4
        self.double_quant = double_quant
        self.model = None
        self.tokenizer = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _create_quantization_config(self):
        """
        创建量化配置
        
        返回:
            BitsAndBytesConfig: 量化配置对象
        """
        if self.quantization_type == "8bit":
            self.logger.info("创建 8 位量化配置")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:  # 4bit
            quant_type = "nf4" if self.nf4 else "fp4"
            self.logger.info(f"创建 4 位量化配置 (类型: {quant_type}, 嵌套量化: {self.double_quant})")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.double_quant,
            )
    
    def quantize(self):
        """
        量化模型
        
        返回:
            self: 返回自身实例以支持链式调用
        """
        self.logger.info(f"开始量化模型: {self.model_name}")
        
        # 创建量化配置
        quantization_config = self._create_quantization_config()
        
        # 加载分词器
        self.logger.info("加载分词器")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # 加载并量化模型
        self.logger.info("加载并量化模型")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 计算模型大小
        model_size_mb = self.model.get_memory_footprint() / (1024 * 1024)
        self.logger.info(f"模型量化完成。模型大小: {model_size_mb:.2f} MB")
        
        return self
    
    def save(self, output_dir):
        """
        保存量化后的模型
        
        参数:
            output_dir (str): 保存模型的目录路径
            
        返回:
            str: 保存模型的路径
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型尚未量化，请先调用 quantize() 方法")
        
        self.logger.info(f"保存量化后的模型到: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存量化配置信息
        config_info = {
            "quantization_type": self.quantization_type,
            "nf4": self.nf4 if self.quantization_type == "4bit" else False,
            "double_quant": self.double_quant if self.quantization_type == "4bit" else False
        }
        
        # 将配置信息写入 README.md
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"# 量化模型: {os.path.basename(self.model_name)}\n\n")
            f.write("## 量化配置\n\n")
            f.write(f"- 量化类型: {self.quantization_type}\n")
            if self.quantization_type == "4bit":
                f.write(f"- 数据类型: {'NF4' if self.nf4 else 'FP4'}\n")
                f.write(f"- 嵌套量化: {'是' if self.double_quant else '否'}\n")
        
        self.logger.info(f"模型已成功保存到: {output_dir}")
        return output_dir
    
    def load_quantized(self, model_path):
        """
        加载已量化的模型
        
        参数:
            model_path (str): 已量化模型的路径
            
        返回:
            self: 返回自身实例以支持链式调用
        """
        self.logger.info(f"加载已量化的模型: {model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.logger.info("已量化模型加载成功")
        return self
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        使用量化模型生成文本
        
        参数:
            prompt (str): 输入提示
            max_new_tokens (int): 生成的最大token数
            temperature (float): 温度参数，控制随机性
            top_p (float): 控制采样的概率阈值
            
        返回:
            str: 生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型尚未加载，请先调用 quantize() 或 load_quantized() 方法")
        
        # 准备输入
        messages = [{"role": "user", "content": prompt}]
        
        # 将消息转换为模型输入格式
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )
        
        # 解码回复
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取模型回复部分
        response = generated_text[len(input_text):].strip()
        
        return response


def main():
    """
    使用示例
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用BitsAndBytes量化模型')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help='要量化的模型名称或路径')
    parser.add_argument('--output_dir', type=str, default="./quantized_model",
                        help='保存量化模型的目录')
    parser.add_argument('--quantization', type=str, choices=['8bit', '4bit'], default='8bit',
                        help='量化方法: 8bit=8位量化, 4bit=4位量化')
    parser.add_argument('--nf4', action='store_true', help='使用NF4数据类型进行4位量化')
    parser.add_argument('--double_quant', action='store_true', help='使用嵌套量化进一步减少内存使用')
    parser.add_argument('--test_prompt', type=str, default="你好，请介绍一下自己。",
                        help='测试提示，用于验证量化后的模型')
    args = parser.parse_args()
    
    # 创建量化器实例
    quantizer = BABQuant(
        model_name=args.model,
        quantization_type=args.quantization,
        nf4=args.nf4,
        double_quant=args.double_quant
    )
    
    # 量化模型
    quantizer.quantize()
    
    # 保存量化后的模型
    output_path = quantizer.save(args.output_dir)
    print(f"量化模型已保存到: {output_path}")
    
    # 测试量化后的模型
    if args.test_prompt:
        print("\n测试量化后的模型:")
        print(f"提示: {args.test_prompt}")
        response = quantizer.generate(args.test_prompt)
        print(f"回复: {response}")


if __name__ == "__main__":
    main() 