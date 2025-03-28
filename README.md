# Qwen2.5-3B 运行示例

这个仓库包含了运行 Qwen/Qwen2.5-3B 大语言模型的示例代码。

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU (推荐至少 8GB 显存)

## 设置虚拟环境

推荐使用虚拟环境来运行此项目，以避免依赖冲突。

### 使用 venv (Python 内置)

```bash
# 创建虚拟环境
python -m venv qwen_env

# 激活虚拟环境
## Linux/macOS
source qwen_env/bin/activate
## Windows
# qwen_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 使用 Conda

```bash
# 创建虚拟环境
conda create -n qwen_env python=3.10

# 激活虚拟环境
conda activate qwen_env

# 安装依赖
pip install -r requirements.txt
```

## 运行模型

确保已激活虚拟环境后，运行以下命令：

```bash
python run_qwen.py
```

## 功能说明

运行脚本后，模型将以对话模式启动：

1. 输入您的问题或指令
2. 模型将生成回复
3. 输入 'exit' 退出对话

## 参数调整

如果您遇到内存不足的问题，可以在 `run_qwen.py` 中修改以下参数：

- 使用 `load_in_8bit=True` 或 `load_in_4bit=True` 来减少内存使用
- 调整 `device_map` 参数
- 减小 `max_new_tokens` 的值

## 注意事项

首次运行时，脚本会自动从 Hugging Face 下载模型权重，这可能需要一些时间，取决于您的网络速度。 

# AWQ 量化指南

## 简介

AWQ（Activation-aware Weight Quantization）是一种高效的模型量化方法，它通过保留对激活值影响较大的权重，实现在低位（如4位）精度下保持模型性能的目标。本指南将帮助您使用 `qwawq.py` 脚本对大型语言模型（如 Qwen 系列）进行 AWQ 量化。

## 安装依赖

在开始之前，请确保安装以下依赖：

```bash
# 安装 AutoAWQ 库
pip install autoawq

# 如果需要使用 ExLlama 版本的 AWQ，请安装最新版本
pip install git+https://github.com/casper-hansen/AutoAWQ.git

# 安装其他依赖
pip install torch transformers
```

## 使用方法

### 基本用法

最简单的用法是使用默认参数进行量化：

```bash
python qwawq.py --model_name_or_path Qwen/Qwen-7B
```

这将使用默认设置（4位量化，group_size=128）对 Qwen-7B 模型进行量化，并将结果保存到 `./qwen_awq` 目录。

### 高级选项

`qwawq.py` 脚本提供了多种参数来自定义量化过程：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen-7B \
  --output_dir ./qwen_awq_custom \
  --bits 4 \
  --group_size 128 \
  --zero_point \
  --version gemm \
  --device cuda:0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 模型路径或 Hugging Face 模型标识符 | Qwen/Qwen-7B |
| `--output_dir` | 保存量化后模型的目录 | ./qwen_awq |
| `--bits` | 量化位数（AWQ 目前仅支持 4 位） | 4 |
| `--group_size` | 量化的分组大小 | 128 |
| `--zero_point` | 是否使用零点量化 | False |
| `--version` | AWQ 版本（"gemm" 或 "exllama"） | gemm |
| `--fuse` | 是否融合 AWQ 模块以提高性能 | False |
| `--fuse_max_seq_len` | 融合模块的最大序列长度 | 2048 |
| `--device` | 加载模型的设备 | cuda:0 |
| `--use_flash_attention` | 是否使用 Flash Attention 2 加速推理 | False |

## 高级用法示例

### 使用 ExLlama 版本加速推理

ExLlama 版本的 AWQ 提供更快的推理速度：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen-7B \
  --version exllama \
  --device cuda:0
```

### 使用融合模块提高性能

融合模块可以提高推理性能和准确性：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen-7B \
  --fuse \
  --fuse_max_seq_len 4096
```

### 使用 Flash Attention 2 加速推理

Flash Attention 2 可以显著提高注意力计算的速度：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen-7B \
  --use_flash_attention
```

注意：融合模块与 Flash Attention 2 不能同时使用。

## 使用量化后的模型

量化完成后，您可以使用以下代码加载和使用量化后的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载量化后的模型和分词器
model_path = "./qwen_awq"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 生成文本
input_text = "今天天气真不错，我想"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## 性能监控

在量化和推理过程中，您可以使用 `nvidia-smi` 命令监控 GPU 资源使用情况：

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

或者在 Python 代码中：

```python
import subprocess

def get_gpu_info():
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

# 使用示例
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu['gpu_id']}: {gpu['memory_used_mb']}MB/{gpu['memory_total_mb']}MB ({gpu['gpu_utilization_percent']}%)")
```

## AWQ 与其他量化方法的比较

| 量化方法 | 支持的位宽 | 内存节省 | 性能保持 | 推理速度 | 适用场景 |
|---------|-----------|---------|---------|---------|---------|
| AWQ     | 主要是 4 位 | ~4倍    | 极好    | 快      | 需要高性能的 4 位量化 |
| GPTQ    | 2-8 位    | 2-8倍   | 良好    | 快      | 需要更灵活位宽选择 |
| GGUF    | 2-8 位    | 2-8倍   | 良好    | 中      | 跨平台兼容性 |
| 量化感知训练 | 8 位    | ~2倍    | 最佳    | 中      | 可接受重新训练 |

## 常见问题

### 1. AWQ 与 GPTQ 的区别是什么？

AWQ 是一种激活感知的权重量化方法，它通过识别和保留对激活值影响较大的权重，在低位精度下保持模型性能。与 GPTQ 相比，AWQ 在 4 位量化时通常能保持更好的模型性能，但目前主要支持 4 位量化，而 GPTQ 可以支持 2-8 位量化。

### 2. 为什么我的 AWQ 量化模型加载失败？

确保您已正确安装 `autoawq` 库，并且使用的是兼容的 Transformers 版本。某些版本的 AutoAWQ 可能会将 Transformers 降级到特定版本（如 4.47.1）。

### 3. 如何在 CPU 上使用 AWQ 量化模型？

AWQ 量化模型主要设计用于 GPU 推理。如果需要在 CPU 上运行，请考虑使用其他量化方法，如 GGUF 或 ONNX 量化。

### 4. 融合模块和 Flash Attention 2 可以一起使用吗？

不可以。融合模块和 Flash Attention 2 是两种不同的优化技术，它们不能同时使用。如果同时指定，脚本会自动禁用 Flash Attention 2。

### 5. AWQ 支持 2 位或 3 位量化吗？

目前，AWQ 主要支持 4 位量化。如果需要更低位宽的量化，建议考虑使用 GPTQ 等其他量化方法。

## 结论

AWQ 量化是一种高效的模型压缩方法，可以显著减少模型大小和内存占用，同时保持良好的性能。通过本指南中的 `qwawq.py` 脚本，您可以轻松地对大型语言模型进行 AWQ 量化，并根据自己的需求调整各种参数。

希望本指南对您有所帮助！如有任何问题，请随时提问。 