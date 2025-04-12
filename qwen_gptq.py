#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 GPTQModel 库量化 Qwen/Qwen2.5-3B 模型的示例代码
"""

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import time
import subprocess
import os
import argparse

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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用GPTQModel量化运行Qwen2.5-3B模型')
    parser.add_argument('--bits', type=int, choices=[2, 3, 4, 8], default=4,
                        help='量化位数: 2, 3, 4 或 8 位')
    parser.add_argument('--group_size', type=int, default=128,
                        help='量化组大小')
    parser.add_argument('--dataset', type=str, default="c4",
                        help='用于校准的数据集，默认使用"c4"')
    parser.add_argument('--custom_dataset', type=str, default=None,
                        help='自定义校准数据集文件路径，每行一个文本')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='量化时的批处理大小')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存量化模型的路径，如果不指定则不保存')
    args = parser.parse_args()
    
    # 记录加载模型前的GPU使用情况
    print("记录加载模型前的GPU使用情况...")
    before_loading_gpu_info = get_gpu_info()
    before_loading_memory = before_loading_gpu_info[0]['memory_used_mb'] if before_loading_gpu_info else "未知"
    print(f"加载前GPU内存使用: {before_loading_memory} MB")
    
    # 设置模型名称
    model_name = "Qwen/Qwen2.5-3B"
    
    # 打印正在加载的模型信息
    print(f"正在加载模型: {model_name}")
    
    # 配置量化参数
    quant_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=True,
        sym=True,
        true_sequential=True
    )
    
    # 加载校准数据集
    if args.custom_dataset:
        try:
            with open(args.custom_dataset, 'r', encoding='utf-8') as f:
                dataset = [line.strip() for line in f if line.strip()]
            print(f"已加载自定义数据集，共 {len(dataset)} 条文本")
        except Exception as e:
            print(f"加载自定义数据集时出错: {e}")
            print("将使用默认数据集")
            dataset = args.dataset
    else:
        print(f"加载数据集: {args.dataset}")
        dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(1024))["text"]
    
    # 加载并量化模型
    print(f"开始使用GPTQ进行{args.bits}位量化...")
    print("这可能需要一些时间，请耐心等待...")
    load_start_time = time.time()
    
    model = GPTQModel.load(model_name, quant_config)
    model.quantize(dataset, batch_size=args.batch_size)
    
    load_time = time.time() - load_start_time
    
    # 设置生成参数
    generation_config = {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    
    # 构建模型描述
    model_description = f"GPTQ {args.bits}位量化模型 (group_size={args.group_size})"
    
    # 运行对话循环
    print("\n" + "="*50)
    print(f"Qwen2.5-3B {model_description}已加载完成。自动开始性能评估测试。")
    print(f"模型加载和量化时间: {load_time:.2f} 秒")
    print("="*50 + "\n")
    
    # 打印模型内存占用
    model_size = model.get_memory_footprint() / (1024 * 1024)  # 转换为MB
    print(f"模型内存占用: {model_size:.2f} MB")
    
    # 性能评估
    test_question = "如何计算矩阵的行列式？"
    print(f"开始性能评估，将连续询问5次: '{test_question}'")
    print("测试进行中，请稍候...\n")
    
    total_inference_time = 0
    total_tokens_generated = 0
    
    for i in range(5):
        # 记录推理开始时间
        inference_start_time = time.time()
        
        # 生成回复
        tokens = model.generate(test_question, **generation_config)[0]
        result = model.tokenizer.decode(tokens)
        
        # 计算推理时间
        inference_time = time.time() - inference_start_time
        
        # 计算生成的令牌数
        new_tokens_count = len(tokens)
        total_tokens_generated += new_tokens_count
        
        # 计算每秒生成的令牌数
        tokens_per_second = new_tokens_count / inference_time
        
        total_inference_time += inference_time
        
        # 显示进度
        print(f"完成测试 {i+1}/5 - 生成了 {new_tokens_count} 个令牌，速度: {tokens_per_second:.2f} 令牌/秒", end="\r")
    
    # 计算平均推理时间和令牌生成速度
    avg_inference_time = total_inference_time / 5
    avg_tokens_per_second = total_tokens_generated / total_inference_time
    avg_tokens_generated = total_tokens_generated / 5
    
    # 打印汇总结果
    print("\n" + "="*50)
    print("性能评估结果汇总:")
    print(f"模型类型: {model_description}")
    print(f"加载前GPU内存使用: {before_loading_memory} MB")
    print(f"模型内存占用: {model_size:.2f} MB")
    print(f"平均推理时间: {avg_inference_time:.2f} 秒")
    print(f"平均生成令牌数: {avg_tokens_generated:.1f} 个")
    print(f"平均生成速度: {avg_tokens_per_second:.2f} 令牌/秒")
    print("="*50 + "\n")
    
    # 保存量化模型
    if args.save_path:
        print(f"正在保存模型到 {args.save_path}...")
        model.save(args.save_path)
        print(f"模型已保存到 {args.save_path}")
    
    # 添加交互式对话模式
    print("\n" + "="*50)
    print("测试完成后，您可以继续进行对话。输入'exit'退出。")
    print("="*50 + "\n")
    
    while True:
        user_input = input("用户: ")
        if user_input.lower() == 'exit':
            print("再见！")
            break
        
        inference_start_time = time.time()
        tokens = model.generate(user_input, **generation_config)[0]
        result = model.tokenizer.decode(tokens)
        inference_time = time.time() - inference_start_time
        
        print(f"Qwen: {result}\n")
        print(f"推理时间: {inference_time:.2f} 秒，生成了 {len(tokens)} 个令牌")

if __name__ == "__main__":
    main() 