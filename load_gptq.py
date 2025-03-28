#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加载本地已量化的 GPTQ 模型并执行性能分析
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
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

def parse_args():
    parser = argparse.ArgumentParser(description='加载本地GPTQ量化模型并执行性能分析')
    parser.add_argument('--model_path', type=str, required=True,
                        help='本地已量化的GPTQ模型路径')
    parser.add_argument('--bits', type=int, default=3,
                        help='模型的量化位宽，默认3位')
    parser.add_argument('--use_exllama', action='store_true',
                        help='是否使用ExLlama后端进行推理')
    parser.add_argument('--exllama_version', type=int, choices=[1, 2], default=2,
                        help='使用的ExLlama版本，1或2，默认为2')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='加载模型的设备')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存性能测试结果')
    parser.add_argument('--text_generation_config', type=str, default=None,
                        help='Text Generation配置文件路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 '{args.model_path}' 不存在！")
        print("请提供正确的本地GPTQ模型路径。")
        return
    
    # 创建结果目录
    results_dir = "performance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 记录加载模型前的GPU使用情况
    print("记录加载模型前的GPU使用情况...")
    before_loading_gpu_info = get_gpu_info()
    before_loading_memory = before_loading_gpu_info[0]['memory_used_mb'] if before_loading_gpu_info else "未知"
    print(f"加载前GPU内存使用: {before_loading_memory} MB")
    
    # 打印正在加载的模型信息
    print(f"正在加载本地GPTQ量化模型: {args.model_path}")
    
    # 加载分词器
    try:
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print("分词器加载成功")
    except Exception as e:
        print(f"加载分词器出错: {e}")
        print("尝试不使用trust_remote_code...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            print("分词器加载成功（无trust_remote_code）")
        except Exception as e2:
            print(f"再次尝试加载分词器失败: {e2}")
            print("请检查模型路径是否包含tokenizer_config.json和tokenizer.json文件")
            return
    
    # 准备加载参数
    load_kwargs = {
        "device_map": args.device,
        "trust_remote_code": True,
    }
    
    # 为ExLlama添加配置
    if args.use_exllama:
        print(f"使用ExLlama v{args.exllama_version}后端...")
        quantization_config = GPTQConfig(
            bits=args.bits,
            exllama_config={"version": args.exllama_version}
        )
        load_kwargs["quantization_config"] = quantization_config
    
    # 记录加载开始时间
    load_start_time = time.time()
    
    # 加载模型
    try:
        print("加载模型中，请耐心等待...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **load_kwargs
        )
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("\n可能的原因:")
        print("1. 模型路径不正确")
        print("2. 模型不是GPTQ格式或格式不兼容")
        print("3. 内存不足")
        print("4. GPU驱动或CUDA版本不兼容")
        print("\n解决方案:")
        print("1. 确保模型路径包含模型配置文件(config.json)和权重文件")
        print("2. 确保模型已经通过GPTQ正确量化")
        print("3. 检查量化位宽是否与模型匹配（默认为3位，可通过--bits参数修改）")
        print("4. 尝试不使用ExLlama后端（不添加--use_exllama参数）")
        print("5. 确保有足够的GPU内存（或尝试使用CPU）")
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
    
    # 如果提供了text generation配置文件，则覆盖默认设置
    if args.text_generation_config:
        try:
            with open(args.text_generation_config, 'r') as f:
                custom_config = json.load(f)
                generation_config.update(custom_config)
                print(f"已加载自定义生成配置: {args.text_generation_config}")
        except Exception as e:
            print(f"加载生成配置文件出错: {e}")
    
    # 构建模型描述
    model_description = f"GPTQ {args.bits}位量化模型"
    if args.use_exllama:
        model_description += f" (ExLlama v{args.exllama_version})"
    
    # 运行对话循环
    print("\n" + "="*50)
    print(f"{args.model_path} {model_description}已加载完成。开始性能评估测试。")
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
    total_tokens_generated = 0
    final_gpu_info = None
    
    for i in range(5):
        # 准备输入
        messages = [{"role": "user", "content": test_question}]
        
        # 将消息转换为模型输入格式
        try:
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"应用聊天模板时出错: {e}")
            print("尝试直接使用问题文本...")
            input_text = test_question
        
        # 编码输入
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        
        # 记录推理开始时间
        inference_start_time = time.time()
        
        # 生成回复
        with torch.no_grad():
            try:
                generated_ids = model.generate(
                    **model_inputs,
                    **generation_config
                )
            except Exception as e:
                print(f"生成回复时出错: {e}")
                print("尝试使用更简单的生成参数...")
                simple_config = {"max_new_tokens": 128}
                generated_ids = model.generate(
                    **model_inputs,
                    **simple_config
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
        results_file = os.path.join(results_dir, f"gptq_load_performance_test_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_path": args.model_path,
                "quantization_method": "GPTQ",
                "quantization_config": {
                    "bits": args.bits,
                    "exllama": args.use_exllama,
                    "exllama_version": args.exllama_version if args.use_exllama else None
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
        try:
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"应用聊天模板时出错: {e}")
            print("直接使用用户输入...")
            input_text = user_input
        
        # 编码输入
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        
        # 记录推理开始时间
        inference_start_time = time.time()
        
        # 生成回复
        with torch.no_grad():
            try:
                generated_ids = model.generate(
                    **model_inputs,
                    **generation_config
                )
            except Exception as e:
                print(f"生成回复时出错: {e}")
                print("尝试使用更简单的生成参数...")
                simple_config = {"max_new_tokens": 128}
                generated_ids = model.generate(
                    **model_inputs,
                    **simple_config
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
        print(f"模型: {response}\n")
        print(f"推理时间: {inference_time:.2f} 秒，生成了 {new_tokens_count} 个令牌，速度: {tokens_per_second:.2f} 令牌/秒")

if __name__ == "__main__":
    main()