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
python qwawq.py
```

这将使用默认设置对 Qwen/Qwen2.5-3B 模型进行 AWQ 4位量化，并将结果保存到 `./qwen_awq` 目录。

### 高级选项

`qwawq.py` 脚本提供了多种参数来自定义量化过程：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --output_dir ./qwen_awq_custom \
  --group_size 128 \
  --zero_point \
  --version gemm \
  --device cuda:0 \
  --save_results
```
python qwawq.py --model_name_or_path /home/lxrobotlab-4090-a/wmy/qwen/gptq-4bit --zero_point --version exllama

python qwawq.py --model_name_or_path /home/lxrobotlab-4090-a/wmy/qwen/gptq-4bit --zero_point --version gemm

python qwawq.py --model_name_or_path /home/lxrobotlab-4090-a/wmy/qwen/gptq-4bit --zero_point --version exllama --fuse


### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 模型路径或 Hugging Face 模型标识符 | Qwen/Qwen2.5-3B |
| `--output_dir` | 保存量化后模型的目录 | ./qwen_awq |
| `--group_size` | 量化的分组大小 | 128 |
| `--zero_point` | 是否使用零点量化 | False |
| `--version` | AWQ 版本（"gemm" 或 "exllama"） | gemm |
| `--fuse` | 是否融合 AWQ 模块以提高性能 | False |
| `--fuse_max_seq_len` | 融合模块的最大序列长度 | 2048 |
| `--device` | 加载模型的设备 | cuda:0 |
| `--use_flash_attention` | 是否使用 Flash Attention 2 加速推理 | False |
| `--save_results` | 是否保存性能测试结果 | False |

## 高级用法示例

### 使用 ExLlama 版本加速推理

ExLlama 版本的 AWQ 提供更快的推理速度：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --version exllama \
  --device cuda:0
```

### 使用融合模块提高性能

融合模块可以提高推理性能和准确性：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --fuse \
  --fuse_max_seq_len 4096
```

### 使用 Flash Attention 2 加速推理

Flash Attention 2 可以显著提高注意力计算的速度：

```bash
python qwawq.py \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --use_flash_attention
```

注意：融合模块与 Flash Attention 2 不能同时使用。

### 保存性能测试结果

如果您想保存性能测试结果以便后续分析：

```bash
python qwawq.py --save_results
```

这将在 `performance_results` 目录中创建一个包含详细测试数据的 JSON 文件。

## 脚本功能

`qwawq.py` 脚本执行以下操作：

1. **加载和量化模型**：使用 AWQ 方法加载并量化指定的模型
2. **保存量化后的模型**：将量化后的模型和分词器保存到指定目录
3. **性能评估**：对模型进行性能测试，连续询问 5 次相同的问题
4. **交互式对话**：测试完成后，允许用户与模型进行交互式对话

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

在量化和推理过程中，脚本会自动监控 GPU 资源使用情况，并在测试结束后显示汇总信息，包括：

- 加载前 GPU 内存使用
- 推理后 GPU 内存使用
- 模型内存占用
- 平均推理时间

如果您想在自己的代码中监控 GPU 资源，可以使用以下函数：

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
| BitsAndBytes | 8/4 位 | 2-4倍  | 良好    | 中      | 简单实现，易于使用 |
| GGUF    | 2-8 位    | 2-8倍   | 良好    | 中      | 跨平台兼容性 |

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

### 6. 如何退出交互式对话模式？

在对话模式中输入 'exit' 即可退出程序。

## 结论

AWQ 量化是一种高效的模型压缩方法，可以显著减少模型大小和内存占用，同时保持良好的性能。通过本指南中的 `qwawq.py` 脚本，您可以轻松地对 Qwen 系列模型进行 AWQ 量化，并评估量化后模型的性能。

希望本指南对您有所帮助！如有任何问题，请随时提问。