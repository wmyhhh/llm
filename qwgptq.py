#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 GPTQ 量化 Qwen/Qwen2.5-3B 模型的示例代码
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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用GPTQ量化运行Qwen2.5-3B模型')
    parser.add_argument('--bits', type=int, choices=[2, 3, 4, 8], default=4,
                        help='量化位数: 2, 3, 4 或 8 位')
    parser.add_argument('--dataset', type=str, default="c4",
                        help='用于校准的数据集，默认使用"c4"')
    parser.add_argument('--use_marlin', action='store_true', 
                        help='使用Marlin后端（仅适用于4位量化和NVIDIA A100 GPU）')
    parser.add_argument('--use_exllama', action='store_true',
                        help='使用ExLlama后端（仅适用于4位量化）')
    parser.add_argument('--exllama_version', type=int, choices=[1, 2], default=2,
                        help='ExLlama版本，1或2，默认为2')
    parser.add_argument('--custom_dataset', type=str, default=None,
                        help='自定义校准数据集文件路径，每行一个文本')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存性能测试结果')
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
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 配置GPTQ量化参数
    if args.custom_dataset:
        # 加载自定义数据集
        try:
            with open(args.custom_dataset, 'r', encoding='utf-8') as f:
                dataset = [line.strip() for line in f if line.strip()]
            print(f"已加载自定义数据集，共 {len(dataset)} 条文本")
        except Exception as e:
            print(f"加载自定义数据集时出错: {e}")
            print("将使用默认数据集")
            dataset = args.dataset
    else:
        dataset = args.dataset
    
    # 配置GPTQ
    gptq_config_kwargs = {
        "bits": args.bits,
        "dataset": dataset,
        "tokenizer": tokenizer
    }
    
    # 配置后端
    if args.bits == 4:
        if args.use_marlin:
            gptq_config_kwargs["backend"] = "marlin"
            print("使用Marlin后端进行4位量化")
        elif args.use_exllama:
            gptq_config_kwargs["exllama_config"] = {"version": args.exllama_version}
            print(f"使用ExLlama v{args.exllama_version}后端进行4位量化")
    
    gptq_config = GPTQConfig(**gptq_config_kwargs)
    
    # 加载并量化模型
    print(f"开始使用GPTQ进行{args.bits}位量化...")
    print("这可能需要一些时间，请耐心等待...")
    load_start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=gptq_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    load_time = time.time() - load_start_time
    
    # 设置生成参数
    generation_config = {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    
    # 构建模型描述
    model_description = f"GPTQ {args.bits}位量化模型"
    if args.bits == 4 and args.use_marlin:
        model_description += " (Marlin后端)"
    elif args.bits == 4 and args.use_exllama:
        model_description += f" (ExLlama v{args.exllama_version}后端)"
    
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
        # 创建结果目录
        results_dir = "performance_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"gptq_performance_test_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": model_name,
                "quantization_method": "GPTQ",
                "quantization_config": {
                    "bits": args.bits,
                    "dataset": str(dataset),
                    "backend": "marlin" if args.use_marlin else "exllama" if args.use_exllama else "auto",
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
    
    # 询问是否保存量化模型
    save_model = input("是否保存量化模型? (y/n): ").lower().strip() == 'y'
    
    if save_model:
        save_path = f"qwen2.5-3b-gptq-{args.bits}bit"
        print(f"正在保存模型到 {save_path}...")
        
        # 如果使用了device_map，需要先将模型移动到CPU
        model.to("cpu")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"模型已保存到 {save_path}")
        print("您可以使用以下代码加载保存的模型:")
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{save_path}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{save_path}")
        """)
    
    # 添加交互式对话模式
    print("\n" + "="*50)
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