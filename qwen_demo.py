def setup_qwen_model(model_name: str = "Qwen/Qwen2.5-3B") -> tuple:
    """
    Setup Qwen model and tokenizer with 8-bit quantization and CPU offloading
    使用8位量化和CPU卸载技术设置Qwen模型和分词器
    
    关键功能:
    1. 8位量化减少内存使用
    2. CPU卸载分散内存压力
    3. 自动设备映射优化资源利用
    4. 性能监控和错误处理
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # 初始化分词器 - 用于处理输入文本
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 设置量化配置
    # - load_in_8bit=True: 启用8位量化，减少内存使用
    # - llm_int8_enable_fp32_cpu_offload: 允许将部分计算卸载到CPU
    # - llm_int8_threshold: 量化阈值，控制量化精度
    # - llm_int8_has_fp16_weight: 保持权重为fp16以提高精度
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True
    )
    
    # 定义模型组件的设备映射
    # - embed_tokens: 词嵌入层放在GPU (设备0)
    # - layers: 主要计算层放在GPU
    # - norm: 归一化层放在GPU
    # - lm_head: 语言模型头部放在CPU（减少GPU内存使用）
    device_map = {
        "embed_tokens": 0,
        "layers": 0,
        "norm": 0,
        "lm_head": "cpu"
    }
    
    # 获取初始系统状态 - 用于比较模型加载前后的资源使用情况
    initial_info = get_system_info()
    print("\n初始系统状态:")
    print(f"CPU 使用率: {initial_info['cpu_percent']}%")
    print(f"内存使用率: {initial_info['ram_usage_percent']}%")
    print(f"进程内存占用: {initial_info['process_memory_mb']:.2f} MB")
    print(f"GPU 内存使用: {initial_info['gpu_memory_used_mb']:.2f} MB / {initial_info['gpu_memory_total_mb']:.2f} MB")
    print(f"GPU 使用率: {initial_info['gpu_utilization']:.2f}%")
    
    # 加载模型 - 使用try-except处理可能的错误
    load_start_time = time.time()
    try:
        # 尝试使用量化和设备映射加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,  # 允许使用模型特定的代码
            quantization_config=quantization_config,  # 应用量化设置
            device_map=device_map,  # 应用设备映射
            torch_dtype="auto"  # 自动选择最适合的数据类型
        )
        print("\n成功加载量化模型并启用 CPU 卸载")
    except Exception as e:
        # 如果量化加载失败，回退到基本的FP16加载
        print(f"\n警告：量化加载失败，尝试使用默认设置: {str(e)}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",  # 自动设备映射
            torch_dtype=torch.float16,  # 使用FP16
            low_cpu_mem_usage=True  # 优化CPU内存使用
        )
    
    # 打印模型组件的内存分布情况
    print("\n模型组件分布:")
    if hasattr(model, 'hf_device_map'):
        for name, device in model.hf_device_map.items():
            print(f"{name}: {device}")
    else:
        print("模型未使用设备映射")
    
    # 验证模型的数据类型
    print("\n模型量化信息:")
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            last_layer = model.model.layers[-1]
            if hasattr(last_layer, 'input_layernorm'):
                dtype = last_layer.input_layernorm.weight.dtype
                print(f"模型最后一层数据类型: {dtype}")
    except Exception as e:
        print(f"无法获取模型数据类型信息: {str(e)}")
    
    # 计算加载时间
    load_time = time.time() - load_start_time
    
    # 获取加载后的系统状态 - 用于评估模型对系统资源的影响
    post_load_info = get_system_info()
    print("\n模型加载后系统状态:")
    print(f"模型加载时间: {load_time:.2f} 秒")
    print(f"CPU 使用率: {post_load_info['cpu_percent']}%")
    print(f"内存使用率: {post_load_info['ram_usage_percent']}%")
    print(f"进程内存占用: {post_load_info['process_memory_mb']:.2f} MB")
    print(f"GPU 内存使用: {post_load_info['gpu_memory_used_mb']:.2f} MB / {post_load_info['gpu_memory_total_mb']:.2f} MB")
    print(f"GPU 使用率: {post_load_info['gpu_utilization']:.2f}%")
    
    return model, tokenizer 