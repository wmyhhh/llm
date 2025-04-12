#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 GPTQModel 库量化 unsloth/Llama-3.2-1B-Instruct 模型的示例代码
"""

import os
# 设置环境变量以优化性能
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 优化内存分配
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 确保正确的设备顺序
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 避免下载额外组件

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import time
import subprocess
import argparse
import json
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
import os

def fix_rope_scaling_config(model_name):
    """修复 Llama-3 模型的 RoPE 缩放配置"""
    try:
        # 下载并读取模型配置
        config_file = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 检查并修复 rope_scaling 配置
        if 'rope_scaling' in config and not ('type' in config['rope_scaling'] and len(config['rope_scaling']) == 2):
            print(f"修复 {model_name} 的 RoPE 缩放配置")
            # 使用兼容的配置格式
            config['rope_scaling'] = {
                "type": "linear",
                "factor": config['rope_scaling'].get('factor', 2.0)
            }
            
            # 保存修改后的配置
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("配置已修复")
        
    except Exception as e:
        print(f"尝试修复配置时出错: {e}")
        print("将在加载时尝试其他方法")

def get_gpu_info():
    """获取 GPU 使用信息"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                           stdout=subprocess.PIPE, 
                           universal_newlines=True)
    gpu_info = result.stdout.strip().split(',')
    return {
        'gpu_util': float(gpu_info[0].strip()),
        'memory_used': float(gpu_info[1].strip()),
        'memory_total': float(gpu_info[2].strip()),
        'memory_percent': float(gpu_info[1].strip()) / float(gpu_info[2].strip()) * 100
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用 GPTQ 量化 unsloth/Llama-3.2-1B-Instruct 模型")
    parser.add_argument('--bits', type=int, default=4, help='量化位数 (默认: 4)')
    parser.add_argument('--group_size', type=int, default=128, help='量化组大小 (默认: 128)')
    parser.add_argument('--dataset', type=str, default="c4", help='校准数据集 (默认: c4)')
    parser.add_argument('--custom_dataset', type=str, default=None, help='自定义校准数据集路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小 (默认: 1)')
    parser.add_argument('--save_path', type=str, default=None, help='保存量化模型的路径')
    args = parser.parse_args()
    
    # 设置模型名称
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    
    # 修复 RoPE 缩放配置
    fix_rope_scaling_config(model_name)
    
    # 获取和打印设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    
    # 记录开始时的 GPU 内存使用
    if torch.cuda.is_available():
        print("加载模型前 GPU 状态:")
        gpu_info_before = get_gpu_info()
        print(f"GPU 使用率: {gpu_info_before['gpu_util']}%")
        print(f"已用显存: {gpu_info_before['memory_used']:.2f} MB / {gpu_info_before['memory_total']:.2f} MB ({gpu_info_before['memory_percent']:.2f}%)")
    
    # 准备量化配置
    print("\n配置量化参数...")
    quant_config = QuantizeConfig(
        bits=args.bits,                     # 量化位数
        group_size=args.group_size,         # 组大小
        desc_act=True,                      # 是否使用描述激活
        act_order=True,                     # 激活排序
        static_groups=False,                # 是否使用静态组
        sym=True,                           # 对称量化
        true_sequential=True                # 是否使用真正的顺序量化
    )
    print(f"量化参数: {quant_config}")
    
    # 加载校准数据集
    print("\n加载校准数据集...")
    if args.custom_dataset:
        # 加载自定义数据集
        with open(args.custom_dataset, 'r') as f:
            calibration_data = [line.strip() for line in f.readlines()]
        print(f"已加载自定义数据集，共 {len(calibration_data)} 条样本")
    else:
        # 加载默认数据集
        dataset = load_dataset(args.dataset, split="train")
        calibration_data = [item["text"] for item in dataset.select(range(100))]
        print(f"已加载 {args.dataset} 数据集，共 {len(calibration_data)} 条样本")
    
    try:
        # 加载和量化模型
        print("\n加载模型...")
        model = GPTQModel.load(model_name, quant_config, attn_implementation="eager")
        
        # 记录加载后的 GPU 内存使用
        if torch.cuda.is_available():
            print("\n加载模型后 GPU 状态:")
            gpu_info_after = get_gpu_info()
            print(f"GPU 使用率: {gpu_info_after['gpu_util']}%")
            print(f"已用显存: {gpu_info_after['memory_used']:.2f} MB / {gpu_info_after['memory_total']:.2f} MB ({gpu_info_after['memory_percent']:.2f}%)")
            print(f"显存增加: {gpu_info_after['memory_used'] - gpu_info_before['memory_used']:.2f} MB")
        
        # 量化模型
        print("\n开始量化模型...")
        start_time = time.time()
        model.quantize(calibration_data, batch_size=args.batch_size)
        end_time = time.time()
        print(f"量化完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 性能评估
        print("\n性能评估...")
        test_prompt = "Tell me a short story about a robot learning to paint."
        
        # 记录推理时间
        start_time = time.time()
        output = model.generate(test_prompt, max_new_tokens=50)
        end_time = time.time()
        
        inference_time = end_time - start_time
        tokens_generated = len(output.split()) - len(test_prompt.split())
        
        print(f"\n测试提示: {test_prompt}")
        print(f"生成输出: {output}")
        print(f"推理时间: {inference_time:.2f} 秒")
        print(f"生成令牌数: {tokens_generated}")
        print(f"生成速度: {tokens_generated / inference_time:.2f} 令牌/秒")
        
        # 保存模型
        if args.save_path:
            print(f"\n保存量化模型到 {args.save_path}...")
            model.save(args.save_path)
            print("模型保存完成")
        
        # 交互式对话
        print("\n进入交互式对话模式，输入 'exit' 退出")
        while True:
            user_input = input("\n请输入提示 > ")
            if user_input.lower() == 'exit':
                break
            
            start_time = time.time()
            response = model.generate(user_input, max_new_tokens=100)
            end_time = time.time()
            
            print(f"\n输出: {response}")
            print(f"生成时间: {end_time - start_time:.2f} 秒")
    
    except Exception as e:
        print(f"错误: {e}")
        # 尝试使用 eager 模式加载模型
        try:
            print("\n尝试使用 eager 模式加载模型...")
            # 尝试加载时修改配置
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'rope_scaling'):
                if not ('type' in config.rope_scaling and len(config.rope_scaling) == 2):
                    config.rope_scaling = {"type": "linear", "factor": 2.0}
                    print("已在内存中修复配置")
            
            model = GPTQModel.load(
                model_name, 
                quant_config, 
                attn_implementation="eager", 
                config=config
            )
            print("成功加载模型")
            
            # 继续执行后续代码...
            # 量化模型
            print("\n开始量化模型...")
            model.quantize(calibration_data, batch_size=args.batch_size)
            print("量化完成")
            
            # 如果需要保存模型
            if args.save_path:
                print(f"\n保存量化模型到 {args.save_path}...")
                model.save(args.save_path)
                print("模型保存完成")
            
        except Exception as e2:
            print(f"二次尝试失败: {e2}")
            print("建议更新 transformers 和 gptqmodel 库到最新版本")

if __name__ == "__main__":
    main() 