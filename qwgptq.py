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
    parser = argparse.ArgumentParser(description="Run Qwen2.5-3B model with GPTQ quantization")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B", 
                        help="Path to the model or model identifier from huggingface.co/models")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], 
                        help="Number of bits for GPTQ quantization")
    parser.add_argument("--group_size", type=int, default=128, 
                        help="Group size for GPTQ quantization")
    parser.add_argument("--dataset", type=str, default="c4", 
                        help="Dataset to use for calibration")
    parser.add_argument("--save_path", type=str, default="", 
                        help="Path to save the quantized model (if empty, model won't be saved)")
    parser.add_argument("--question", type=str, default="请介绍一下自己", 
                        help="Question to ask the model")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "marlin", "exllama"], 
                        help="Backend to use for GPTQ quantization")
    parser.add_argument("--disable_exllama", action="store_true", 
                        help="Disable exllama backend")
    parser.add_argument("--disable_marlin", action="store_true", 
                        help="Disable marlin backend")
    
    args = parser.parse_args()
    
    # Print arguments
    logger.info(f"Running with arguments: {args}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. GPTQ quantization requires a GPU.")
        return
    
    # Determine the backend to use
    backend = args.backend
    if backend == "auto":
        if args.disable_exllama and args.disable_marlin:
            backend = None
        elif args.disable_exllama:
            backend = "marlin"
        elif args.disable_marlin:
            backend = "exllama"
        else:
            # Try to determine the best backend based on availability
            try:
                import exllama
                backend = "exllama"
                logger.info("Using ExLlama backend")
            except ImportError:
                try:
                    import marlin
                    backend = "marlin"
                    logger.info("Using Marlin backend")
                except ImportError:
                    backend = None
                    logger.info("No specialized backend available, using default")
    
    # Create GPTQ configuration
    from transformers import GPTQConfig
    
    # If using Marlin backend, we need to handle it differently due to the pack attribute issue
    if backend == "marlin":
        logger.info("Using Marlin backend with special handling")
        # For Marlin backend, we'll use a two-step process:
        # 1. First quantize with a temporary directory
        # 2. Then load the quantized model with the correct backend
        
        # Step 1: Quantize with a temporary directory
        temp_save_path = "./temp_quantized_model"
        os.makedirs(temp_save_path, exist_ok=True)
        
        # Use a compatible backend for quantization
        quantization_config = GPTQConfig(
            bits=args.bits,
            group_size=args.group_size,
            dataset=args.dataset,
            # Don't specify backend here to avoid the pack issue
        )
    else:
        # For other backends, proceed normally
        quantization_config = GPTQConfig(
            bits=args.bits,
            group_size=args.group_size,
            dataset=args.dataset,
            use_exllama=backend == "exllama"
        )
    
    logger.info(f"GPTQ Configuration: {quantization_config}")
    
    # Record memory before loading the model
    initial_memory = get_memory_usage()
    initial_gpu_info = get_gpu_info()
    logger.info(f"Initial CPU Memory Usage: {initial_memory:.2f} MB")
    logger.info(f"Initial GPU Memory Usage: {initial_gpu_info}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Special handling for Marlin backend
    if backend == "marlin":
        # First, quantize without specifying the backend
        logger.info("Step 1: Quantizing model without Marlin backend...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
        
        # Save the quantized model to a temporary location
        logger.info(f"Saving temporarily quantized model to {temp_save_path}")
        model.save_pretrained(temp_save_path)
        tokenizer.save_pretrained(temp_save_path)
        
        # Clear CUDA cache and delete the model to free memory
        del model
        torch.cuda.empty_cache()
        
        # Now load the model with Marlin backend
        logger.info("Step 2: Loading quantized model with Marlin backend...")
        try:
            # Create a new config with Marlin backend
            marlin_config = GPTQConfig(
                bits=args.bits,
                group_size=args.group_size,
                use_marlin=True
            )
            
            # Load the quantized model with Marlin backend
            model = AutoModelForCausalLM.from_pretrained(
                temp_save_path,
                quantization_config=marlin_config,
                device_map="auto",
            )
        except Exception as e:
            logger.error(f"Error loading with Marlin backend: {e}")
            logger.info("Falling back to default backend...")
            model = AutoModelForCausalLM.from_pretrained(
                temp_save_path,
                device_map="auto",
            )
    else:
        # Normal loading for other backends
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Record memory after loading the model
    post_load_memory = get_memory_usage()
    post_load_gpu_info = get_gpu_info()
    logger.info(f"Post-load CPU Memory Usage: {post_load_memory:.2f} MB")
    logger.info(f"Post-load GPU Memory Usage: {post_load_gpu_info}")
    
    # Calculate model memory footprint
    model_size_mb = model.get_memory_footprint() / (1024 * 1024)
    logger.info(f"Model Memory Footprint: {model_size_mb:.2f} MB")
    
    # Prepare input
    logger.info(f"Preparing to process question: '{args.question}'")
    inputs = tokenizer(args.question, return_tensors="pt").to(model.device)
    
    # Run inference
    logger.info("Running inference...")
    
    # Warm-up run
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
    
    # Measure inference time
    start_inference = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    inference_time = time.time() - start_inference
    
    # Record memory after inference
    post_inference_memory = get_memory_usage()
    post_inference_gpu_info = get_gpu_info()
    logger.info(f"Post-inference CPU Memory Usage: {post_inference_memory:.2f} MB")
    logger.info(f"Post-inference GPU Memory Usage: {post_inference_gpu_info}")
    
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Response: {response}")
    
    # Print performance metrics
    input_length = len(inputs.input_ids[0])
    output_length = len(outputs[0]) - input_length
    tokens_per_second = output_length / inference_time
    
    logger.info(f"Input length: {input_length} tokens")
    logger.info(f"Output length: {output_length} tokens")
    logger.info(f"Inference time: {inference_time:.2f} seconds")
    logger.info(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Memory usage summary
    logger.info("\nMemory Usage Summary:")
    logger.info(f"Initial CPU Memory: {initial_memory:.2f} MB")
    logger.info(f"Post-load CPU Memory: {post_load_memory:.2f} MB")
    logger.info(f"Post-inference CPU Memory: {post_inference_memory:.2f} MB")
    logger.info(f"Model Memory Footprint: {model_size_mb:.2f} MB")
    
    # GPU memory usage
    logger.info("\nGPU Memory Usage Summary:")
    logger.info(f"Initial GPU Memory: {initial_gpu_info}")
    logger.info(f"Post-load GPU Memory: {post_load_gpu_info}")
    logger.info(f"Post-inference GPU Memory: {post_inference_gpu_info}")
    
    # Save the quantized model if a save path is provided
    if args.save_path:
        logger.info(f"Saving quantized model to {args.save_path}")
        # If we used the temporary directory for Marlin, copy from there
        if backend == "marlin" and os.path.exists(temp_save_path):
            import shutil
            if os.path.exists(args.save_path):
                shutil.rmtree(args.save_path)
            shutil.copytree(temp_save_path, args.save_path)
            logger.info(f"Copied quantized model from {temp_save_path} to {args.save_path}")
        else:
            # Otherwise save directly
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
        logger.info("Model and tokenizer saved successfully")
    
    # Clean up temporary directory if it exists
    if backend == "marlin" and os.path.exists(temp_save_path):
        import shutil
        shutil.rmtree(temp_save_path)
        logger.info(f"Cleaned up temporary directory {temp_save_path}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 