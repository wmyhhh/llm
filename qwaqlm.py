#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 AQLM 量化 Qwen/Qwen2.5-3B 模型的示例代码
支持不同的 AQLM 配置，包含性能评估和对话模式
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AqlmConfig
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
    parser = argparse.ArgumentParser(description='使用AQLM量化运行Qwen2.5-3B模型')
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen2.5-3B",
                        help='模型路径或Hugging Face模型标识符')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='加载模型的设备')
    parser.add_argument('--codebook_size', type=int, default=16,
                        help='AQLM码本大小，通常为16')
    parser.add_argument('--num_codebooks', type=int, default=1,
                        help='AQLM码本数量，通常为1或2')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存性能测试结果')
    parser.add_argument('--pre_quantized', action='store_true',
                        help='是否加载预量化的模型（AQLM需要预量化模型）')
    parser.add_argument('--group_size', type=int, default=16,
                        help='量化的分组大小')
    parser.add_argument('--use_optimum', action='store_true',
                        help='是否使用Optimum后端')
    parser.add_argument('--output_dir', type=str, default="./qwen_aqlm",
                        help='保存量化后模型的目录')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查是否安装了aqlm
    try:
        import aqlm
    except ImportError:
        raise ImportError(
            "AQLM未安装。请使用 `pip install aqlm[gpu,cpu]` 安装。"
            "注意：AQLM需要Python 3.10或更高版本。"
        )
    
    # 检查配置兼容性
    if args.use_cuda:
        if not ((args.num_codebooks == 1 and args.codebook_size == 16) or 
                (args.num_codebooks == 2 and args.codebook_size == 8)):
            print("警告：CUDA后端仅支持1x16和2x8配置。将使用Triton后端。")
            args.use_cuda = False
            args.use_triton = True
    
    if args.use_numba and args.codebook_size != 8:
        print("警告：Numba后端仅支持Kx8配置。将使用Triton后端。")
        args.use_numba = False
        args.use_triton = True
    
    # 如果没有指定后端，根据配置自动选择
    if not (args.use_triton or args.use_cuda or args.use_numba):
        if args.num_codebooks == 1 and args.codebook_size == 16:
            args.use_cuda = True
            print("自动选择CUDA后端用于1x16配置")
        elif args.num_codebooks == 2 and args.codebook_size == 8:
            args.use_cuda = True
            print("自动选择CUDA后端用于2x8配置")
        elif args.codebook_size == 8:
            if args.device.startswith("cuda"):
                args.use_triton = True
                print("自动选择Triton后端用于GPU上的Kx8配置")
            else:
                args.use_numba = True
                print("自动选择Numba后端用于CPU上的Kx8配置")
        else:
            args.use_triton = True
            print("自动选择Triton后端")
    
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
    
    # 设置模型名称
    model_name = args.model_name_or_path
    
    # 打印正在加载的模型信息
    print(f"正在加载模型: {model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载模型
    print(f"开始使用AQLM进行{args.num_codebooks}x{args.codebook_size}量化...")
    
    # 创建AQLM配置
    aqlm_config = AqlmConfig(
        nbits=args.num_codebooks,
        bits=args.codebook_size,
        group_size=args.group_size,
        use_optimum=args.use_optimum,
    )
    
    # 加载模型参数
    load_kwargs = {
        "quantization_config": aqlm_config,
        "device_map": args.device,
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    
    # 添加pre_quantized参数（AQLM需要预量化模型）
    if args.pre_quantized:
        load_kwargs["pre_quantized"] = True
        print("使用预量化模型模式。")
    else:
        print("警告: AQLM通常需要预量化模型。如果遇到错误，请尝试添加 --pre_quantized 参数。")
    
    # 记录加载开始时间
    load_start_time = time.time()
    
    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
    except ValueError as e:
        if "pre_quantized" in str(e):
            print("\n错误: AQLM需要预量化的模型。")
            print("解决方案: 请使用 --pre_quantized 参数重新运行脚本。")
            print("完整命令示例: python qwaqlm.py --pre_quantized")
            return
        else:
            print(f"加载模型时出错: {e}")
            return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("\n可能的解决方案:")
        print("1. 确保已安装所有必要的依赖: pip install transformers optimum aqlm")
        print("2. 尝试使用预量化模型: 添加 --pre_quantized 参数")
        print("3. 检查模型路径是否正确")
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
    model_description = f"AQLM {args.num_codebooks}x{args.codebook_size}量化模型"
    if args.use_triton:
        model_description += " (Triton后端)"
    elif args.use_cuda:
        model_description += " (CUDA后端)"
    elif args.use_numba:
        model_description += " (Numba后端)"
    
    # 运行对话循环
    print("\n" + "="*50)
    print(f"{model_name} {model_description}已加载完成。自动开始性能评估测试。")
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
        results_file = os.path.join(results_dir, f"aqlm_performance_test_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": model_name,
                "quantization_method": "AQLM",
                "quantization_config": {
                    "num_codebooks": args.num_codebooks,
                    "codebook_size": args.codebook_size,
                    "backend": "triton" if args.use_triton else "cuda" if args.use_cuda else "numba" if args.use_numba else "auto"
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