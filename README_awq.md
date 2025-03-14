# Qwen2.5-3B 模型 AWQ 量化指南

## AWQ 量化简介

AWQ (Activation-aware Weight Quantization) 是一种高效的模型量化方法，专为大型语言模型设计。与其他量化方法相比，AWQ 具有以下优势：

- **更高的精度**：通过识别和保护模型中的"敏感"权重，AWQ 在低位量化（如 4 位）下仍能保持接近原始模型的性能
- **更快的推理速度**：AWQ 针对 GPU 推理进行了优化，提供了专门的 CUDA 内核
- **内存效率**：4 位量化可将模型大小减少约 4 倍，显著降低 GPU 内存需求

本指南将帮助您使用 AWQ 量化 Qwen2.5-3B 模型，并进行高效推理。

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU（推荐至少 8GB VRAM）
- PyTorch 2.0+
- Transformers 4.36.0+
- AutoAWQ 0.1.4+

## 安装依赖

```bash
pip install -r requirements_awq.txt
```

## 基本用法

使用默认参数运行 AWQ 量化（4 位量化）：

```bash
python qwawq.py
```

这将：
1. 加载 Qwen2.5-3B 模型
2. 使用 AWQ 方法将模型量化为 4 位精度
3. 运行一个简单的推理测试
4. 显示性能和内存使用情况统计

## 高级用法

### 自定义量化参数

```bash
python qwawq.py --group_size 64 --zero_point --version exllama
```

### 保存量化后的模型

```bash
python qwawq.py --save_path ./qwen2.5-3b-awq-4bit
```

### 使用自定义问题进行测试

```bash
python qwawq.py --question "解释一下量子计算的基本原理"
```

## 参数说明

- `--model_path`: 模型路径或 Hugging Face 模型标识符（默认：`"Qwen/Qwen2.5-3B"`）
- `--bits`: AWQ 量化的位数（目前仅支持 4 位）
- `--group_size`: 量化的组大小（默认：128）
- `--zero_point`: 是否使用零点量化（默认：不使用）
- `--version`: AWQ 版本，可选 "gemm" 或 "exllama"（默认："gemm"）
- `--do_fuse`: 是否融合 AWQ 模块以提高性能（默认：不融合）
- `--fuse_max_seq_len`: 融合模块的最大序列长度（默认：2048）
- `--save_path`: 保存量化模型的路径（如果为空，则不保存模型）
- `--question`: 用于测试模型的问题（默认："请介绍一下自己"）

## 加载已量化的模型

如果您已经保存了量化后的模型，可以使用以下代码加载它：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载量化后的模型
model = AutoModelForCausalLM.from_pretrained(
    "path/to/quantized/model",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("path/to/quantized/model")

# 使用模型进行推理
inputs = tokenizer("请介绍一下自己", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 性能比较

| 量化方法 | 内存使用 | 推理速度 | 质量损失 |
|---------|---------|---------|---------|
| 原始 (FP16) | ~6GB | 基准 | 无 |
| AWQ (4-bit) | ~1.5GB | 略慢于 FP16 | 极小 |

## 注意事项

1. **首次运行**：首次运行时，脚本将下载 Qwen2.5-3B 模型，这可能需要一些时间，取决于您的网络速度。

2. **量化时间**：AWQ 量化过程可能需要几分钟时间，取决于您的硬件。

3. **版本选择**：
   - `gemm` 版本通常提供更好的通用性能
   - `exllama` 版本在某些硬件上可能提供更快的推理速度，特别是对于长序列

4. **与其他量化方法比较**：
   - 与 GPTQ 相比，AWQ 通常在相同位数下提供更好的性能
   - 与 BitsAndBytes (QLoRA) 相比，AWQ 提供更快的推理速度，但不支持微调

5. **内存使用**：虽然 AWQ 显著减少了模型大小，但在推理过程中仍需要额外的内存用于激活值和注意力缓存。

## 故障排除

如果遇到 `ImportError: No module named 'autoawq'` 错误，请确保已安装 AutoAWQ 库：

```bash
pip install autoawq
```

如果遇到 CUDA 相关错误，请确保您的 PyTorch 版本与您的 CUDA 版本兼容。

## 参考资料

- [Hugging Face AWQ 文档](https://huggingface.co/docs/transformers/quantization/awq)
- [AutoAWQ GitHub 仓库](https://github.com/casper-hansen/AutoAWQ)
- [AWQ 论文：激活感知权重量化](https://arxiv.org/abs/2306.00978) 