#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 AWQ 量化 Qwen/Qwen2.5-3B 模型的示例代码
支持 4-bit 量化，包含性能评估和对话模式
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
import time
import subprocess
import os
import json
import argparse
from datetime import datetime
from transformers.utils import logging

logger = logging.get_logger(__name__)

def get_gpu_info():
    """获取GPU使用情况"""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
        result = result.decode('utf-8').strip()
        lines = result.split('\n')
        gpu_info = []
        for i, line in enumerate(lines):
            memory_used, memory_total, gpu_util = line.split(', ')
            gpu_info.append({
                'gpu_id': i,
                'memory_used_mb': int(memory_used),
                'memory_total_mb': int(memory_total),
                'gpu_utilization_percent': int(gpu_util)
            })
        return gpu_info
    except Exception as e:
        print(f"获取GPU信息时出错: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="使用AWQ量化运行Qwen2.5-3B模型")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="模型路径或Hugging Face模型标识符",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen_awq",
        help="保存量化后模型的目录",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="量化的分组大小",
    )
    parser.add_argument(
        "--zero_point",
        action="store_true",
        help="是否使用零点量化",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="gemm",
        choices=["gemm", "exllama", "ipex"],
        help="AWQ版本: 'gemm'是标准版本, 'exllama'使用ExLlama-v2内核加速推理, 'ipex'用于Intel CPU/GPU",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="是否融合AWQ模块以提高精度和性能",
    )
    parser.add_argument(
        "--fuse_max_seq_len",
        type=int,
        default=2048,
        help="融合模块的最大序列长度，应包括上下文长度和预期生成长度",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="加载模型的设备",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="是否使用Flash Attention 2加速推理（与融合模块不兼容）",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="是否保存性能测试结果",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查选项兼容性
    if args.fuse and args.use_flash_attention:
        logger.warning("融合模块不能与Flash Attention同时使用。禁用Flash Attention。")
        args.use_flash_attention = False
    
    # 检查 Intel 设备支持
    if args.version == "ipex":
        if args.device.startswith("cuda"):
            logger.warning("IPEX 版本不支持 CUDA 设备。将设备更改为 'cpu' 或 'xpu'（Intel GPU）。")
            args.device = "cpu"
        
        try:
            import intel_extension_for_pytorch as ipex
            print("已安装 Intel Extension for PyTorch")
        except ImportError:
            logger.warning(
                "未安装 Intel Extension for PyTorch。请使用 "
                "`pip install intel-extension-for-pytorch` 安装。"
                "对于 IPEX-GPU，请参考 https://intel.github.io/intel-extension-for-pytorch/xpu/2.5.10+xpu/"
            )
    
    # 创建结果目录
    results_dir = "performance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录加载模型前的GPU使用情况
    print("记录加载模型前的GPU使用情况...")
    before_loading_gpu_info = get_gpu_info()
    before_loading_memory = before_loading_gpu_info[0]['memory_used_mb'] if before_loading_gpu_info else "未知"
    print(f"加载前GPU内存使用: {before_loading_memory} MB")
    
    # 打印正在加载的模型信息
    print(f"正在加载模型: {args.model_name_or_path}")
    
    # 创建AWQ配置 - 直接使用transformers的AwqConfig，不依赖autoawq包
    quantization_config = AwqConfig(
        bits=4,  # AWQ目前仅支持4位量化
        group_size=args.group_size,
        zero_point=args.zero_point,
        version=args.version,
    )
    
    # 添加融合模块配置（如果启用）
    if args.fuse:
        quantization_config.do_fuse = True
        quantization_config.fuse_max_seq_len = args.fuse_max_seq_len
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # 加载模型参数
    load_kwargs = {
        "quantization_config": quantization_config,
        "device_map": args.device,
        "trust_remote_code": True,
    }
    
    # 添加 torch_dtype 参数，默认使用 fp16
    load_kwargs["torch_dtype"] = torch.float16
    
    # 添加Flash Attention（如果启用）
    if args.use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    
    # 记录加载开始时间
    load_start_time = time.time()
    
    # 加载模型
    print("加载和量化模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **load_kwargs
        )
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("\n可能的原因:")
        print("1. autoawq 包未正确安装或版本不兼容")
        print("2. transformers 版本可能被 autoawq 降级到 4.47.1")
        print("\n解决方案:")
        print("1. 尝试重新安装 autoawq: pip install -U autoawq")
        print("2. 或安装最新版本: pip install git+https://github.com/casper-hansen/AutoAWQ.git")
        print("3. 安装后重新安装 transformers: pip install -U transformers")
        print("4. 确保 CUDA 和 PyTorch 版本兼容")
        return
    
    # 计算加载时间
    load_time = time.time() - load_start_time
    
    # 设置生成参数
    generation_config = {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    
    # 构建模型描述
    model_description = "AWQ 4位量化模型"
    if args.version == "exllama":
        model_description += " (ExLlama-v2)"
    elif args.version == "ipex":
        model_description += f" (IPEX - {args.device})"
    if args.fuse:
        model_description += " + 融合模块"
    if args.use_flash_attention:
        model_description += " + Flash Attention 2"
    if args.zero_point:
        model_description += " + 零点量化"
    
    # 运行对话循环
    print("\n" + "="*50)
    print(f"{args.model_name_or_path} {model_description}已加载完成。自动开始性能评估测试。")
    print(f"模型加载时间: {load_time:.2f} 秒")
    print("="*50 + "\n")
    
    # 打印模型内存占用
    model_size = model.get_memory_footprint() / (1024 * 1024)  # 转换为MB
    print(f"模型内存占用: {model_size:.2f} MB")
    
    # 保存量化后的模型和分词器
    print(f"保存量化后的模型到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 性能评估
    test_question = "如何计算矩阵的行列式？"
    print(f"开始性能评估，将连续询问5次: '{test_question}'")
    print("测试进行中，请稍候...\n")
    
    performance_results = []
    total_inference_time = 0
    total_tokens_generated = 0
    final_gpu_info = None
    
    for i in range(5):
        # 准备输入
        messages = [{"role": "user", "content": test_question}]
        
        # 将消息转换为模型输入格式
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        
        # 记录推理开始时间
        inference_start_time = time.time()
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **generation_config
            )
        
        # 计算推理时间
        inference_time = time.time() - inference_start_time
        
        # 计算生成的令牌数
        output_token_count = generated_ids.shape[1]
        new_tokens_count = output_token_count - input_token_count
        total_tokens_generated += new_tokens_count
        
        # 计算每秒生成的令牌数
        tokens_per_second = new_tokens_count / inference_time
        
        # 记录推理后的GPU使用情况
        after_inference_gpu_info = get_gpu_info()
        final_gpu_info = after_inference_gpu_info  # 保存最后一次的GPU信息
        
        # 解码回复
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取模型回复部分
        response = generated_text[len(input_text):].strip()
        
        # 保存结果（不打印）
        performance_results.append({
            "test_number": i + 1,
            "question": test_question,
            "response": response,
            "inference_time_seconds": inference_time,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "new_tokens": new_tokens_count,
            "tokens_per_second": tokens_per_second,
            "gpu_info": after_inference_gpu_info
        })
        
        total_inference_time += inference_time
        
        # 显示进度
        print(f"完成测试 {i+1}/5 - 生成了 {new_tokens_count} 个令牌，速度: {tokens_per_second:.2f} 令牌/秒", end="\r")
    
    # 保存性能测试结果
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"awq_performance_test_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": args.model_name_or_path,
                "quantization_method": "AWQ",
                "quantization_config": {
                    "bits": 4,
                    "group_size": args.group_size,
                    "zero_point": args.zero_point,
                    "version": args.version,
                    "fuse": args.fuse,
                    "fuse_max_seq_len": args.fuse_max_seq_len if args.fuse else None,
                    "use_flash_attention": args.use_flash_attention
                },
                "before_loading_gpu_info": before_loading_gpu_info,
                "generation_config": generation_config,
                "test_results": performance_results
            }, f, ensure_ascii=False, indent=2)
    
    # 计算平均推理时间和令牌生成速度
    avg_inference_time = total_inference_time / 5
    avg_tokens_per_second = total_tokens_generated / total_inference_time
    avg_tokens_generated = total_tokens_generated / 5
    
    # 获取最终内存使用情况
    after_inference_memory = final_gpu_info[0]['memory_used_mb'] if final_gpu_info else "未知"
    
    # 打印汇总结果
    print("\n" + "="*50)
    print("性能评估结果汇总:")
    print(f"模型类型: {model_description}")
    print(f"加载前GPU内存使用: {before_loading_memory} MB")
    print(f"推理后GPU内存使用: {after_inference_memory} MB")
    print(f"模型内存占用: {model_size:.2f} MB")
    print(f"平均推理时间: {avg_inference_time:.2f} 秒")
    print(f"平均生成令牌数: {avg_tokens_generated:.1f} 个")
    print(f"平均生成速度: {avg_tokens_per_second:.2f} 令牌/秒")
    if args.save_results:
        print(f"详细结果已保存至: {results_file}")
    print("="*50 + "\n")
    
    print("测试完成后，您可以继续进行对话。输入'exit'退出。")
    print("="*50 + "\n")
    
    # 正常对话模式
    while True:
        # 获取用户输入
        user_input = input("用户: ")
        
        # 检查是否退出
        if user_input.lower() == 'exit':
            print("再见！")
            break
        
        # 准备输入
        messages = [{"role": "user", "content": user_input}]
        
        # 将消息转换为模型输入格式
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        
        # 记录推理开始时间
        inference_start_time = time.time()
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **generation_config
            )
        
        # 计算推理时间
        inference_time = time.time() - inference_start_time
        
        # 计算生成的令牌数
        output_token_count = generated_ids.shape[1]
        new_tokens_count = output_token_count - input_token_count
        
        # 计算每秒生成的令牌数
        tokens_per_second = new_tokens_count / inference_time
        
        # 解码回复
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取模型回复部分
        response = generated_text[len(input_text):].strip()
        
        # 打印回复和性能信息
        print(f"Qwen: {response}\n")
        print(f"推理时间: {inference_time:.2f} 秒，生成了 {new_tokens_count} 个令牌，速度: {tokens_per_second:.2f} 令牌/秒")

if __name__ == "__main__":
    main() 