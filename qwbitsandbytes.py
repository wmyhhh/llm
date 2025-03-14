#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 bitsandbytes 量化 Qwen/Qwen2.5-3B 模型的示例代码
支持 8-bit 和 4-bit 量化
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import subprocess
import os
import json
import argparse
from datetime import datetime

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
    parser = argparse.ArgumentParser(description='使用bitsandbytes量化运行Qwen2.5-3B模型')
    parser.add_argument('--quantization', type=str, choices=['none', '8bit', '4bit'], default='8bit',
                        help='量化方法: none=不量化, 8bit=8位量化, 4bit=4位量化')
    parser.add_argument('--nf4', action='store_true', help='使用NF4数据类型进行4位量化')
    parser.add_argument('--double_quant', action='store_true', help='使用嵌套量化进一步减少内存使用')
    parser.add_argument('--compute_dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='float16',
                        help='计算数据类型')
    args = parser.parse_args()
    
    # 创建结果目录
    results_dir = "performance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 记录加载模型前的GPU使用情况
    print("记录加载模型前的GPU使用情况...")
    before_loading_gpu_info = get_gpu_info()
    before_loading_memory = before_loading_gpu_info[0]['memory_used_mb'] if before_loading_gpu_info else "未知"
    print(f"加载前GPU内存使用: {before_loading_memory} MB")
    
    # 设置模型名称
    model_name = "Qwen/Qwen2.5-3B"
    
    # 打印正在加载的模型信息
    print(f"正在加载模型: {model_name}")
    
    # 设置计算数据类型
    if args.compute_dtype == 'float16':
        compute_dtype = torch.float16
    elif args.compute_dtype == 'bfloat16':
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32
    
    # 配置量化参数
    quantization_config = None
    model_description = "原始模型"
    
    if args.quantization == '8bit':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # 默认阈值
            llm_int8_skip_modules=None,  # 可以跳过某些模块不进行量化
        )
        model_description = "8位量化模型"
    
    elif args.quantization == '4bit':
        quant_type = "nf4" if args.nf4 else "fp4"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,  # 可以是 "fp4" 或 "nf4"
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,  # 是否使用嵌套量化
        )
        model_description = f"4位量化模型 ({quant_type})"
        if args.double_quant:
            model_description += " + 嵌套量化"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载模型
    load_start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,  # 非量化部分的数据类型
        device_map="auto",          # 自动决定模型在哪些设备上运行
        trust_remote_code=True      # 允许运行模型仓库中的自定义代码
    )
    
    load_time = time.time() - load_start_time
    
    # 设置生成参数
    generation_config = {
        "max_new_tokens": 2048,      # 最多生成的新token数量
        "temperature": 0.7,         # 温度参数，控制随机性
        "top_p": 0.9,               # Top-p采样参数
        "repetition_penalty": 1.1,  # 重复惩罚参数
    }
    
    # 运行对话循环
    print("\n" + "="*50)
    print(f"Qwen2.5-3B {model_description}已加载完成。自动开始性能评估测试。")
    print(f"模型加载时间: {load_time:.2f} 秒")
    print("="*50 + "\n")
    
    # 打印模型内存占用
    model_size = model.get_memory_footprint() / (1024 * 1024)  # 转换为MB
    print(f"模型内存占用: {model_size:.2f} MB")
    
    # 性能评估
    test_question = "如何计算矩阵的行列式？"
    print(f"开始性能评估，将连续询问5次: '{test_question}'")
    print("测试进行中，请稍候...\n")
    
    performance_results = []
    total_inference_time = 0
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
        total_inference_time += inference_time
        
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
            "gpu_info": after_inference_gpu_info
        })
        
        # 显示进度
        print(f"完成测试 {i+1}/5", end="\r")
    
    # 保存性能测试结果（但不输出文件路径）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"performance_test_{args.quantization}_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "quantization_type": args.quantization,
            "model_description": model_description,
            "model_size_mb": model_size,
            "before_loading_gpu_info": before_loading_gpu_info,
            "generation_config": generation_config,
            "test_results": performance_results
        }, f, ensure_ascii=False, indent=2)
    
    # 计算平均推理时间
    avg_inference_time = total_inference_time / 5
    
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
        
        # 解码回复
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取模型回复部分
        response = generated_text[len(input_text):].strip()
        
        # 打印回复和性能信息
        print(f"Qwen: {response}\n")
        print(f"推理时间: {inference_time:.2f} 秒")

if __name__ == "__main__":
    main() 