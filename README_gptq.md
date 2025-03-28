# Qwen2.5-3B GPTQ 量化版本运行示例

这个仓库包含了使用 GPTQ 量化运行 Qwen/Qwen2.5-3B 大语言模型的示例代码。

## GPTQ 量化简介

GPTQ 是一种高效的量化方法，可以将模型权重从 FP16/FP32 转换为低位精度（如 4 位、3 位或 2 位），同时保持模型性能。与 bitsandbytes 等其他量化方法相比，GPTQ 具有以下优势：

- **更高的压缩率**：可以实现 2 位、3 位量化，比 8 位量化节省更多内存
- **更好的性能保持**：通过校准过程，最小化量化对模型性能的影响
- **更快的推理速度**：支持 ExLlama 和 Marlin 等高性能后端

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU (推荐至少 8GB 显存)
- PyTorch 2.0+

## 安装依赖

```bash
pip install -r requirements_gptq.txt
```

## 运行模型

### 基本用法

```bash
# 使用默认的 4 位量化
python qwen_gptq.py
```

### 高级用法

```bash
# 使用 2 位量化
python qwen_gptq.py --bits 2

# 使用 3 位量化
python qwen_gptq.py --bits 3

# 使用 4 位量化 + ExLlama v2 后端
python qwen_gptq.py --bits 4 --use_exllama

# 使用 4 位量化 + ExLlama v1 后端
python qwen_gptq.py --bits 4 --use_exllama --exllama_version 1

# 使用 4 位量化 + Marlin 后端 (仅适用于 A100 GPU)
python qwen_gptq.py --bits 4 --use_marlin

# 使用自定义校准数据集
python qwen_gptq.py --custom_dataset my_dataset.txt
```

## 参数说明

- `--bits`: 量化位数 (2, 3, 4, 8)
- `--dataset`: 用于校准的数据集名称，默认为 "c4"
- `--use_exllama`: 使用 ExLlama 后端 (仅适用于 4 位量化)
- `--exllama_version`: ExLlama 版本 (1 或 2)，默认为 2
- `--use_marlin`: 使用 Marlin 后端 (仅适用于 4 位量化和 A100 GPU)
- `--custom_dataset`: 自定义校准数据集文件路径，每行一个文本

## 保存和加载量化模型

运行脚本后，您可以选择保存量化模型。保存后，可以使用以下代码加载模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("qwen2.5-3b-gptq-4bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("qwen2.5-3b-gptq-4bit")
```

## 性能比较

| 量化方法 | 内存使用 | 推理速度 | 质量影响 |
|---------|---------|---------|---------|
| 无量化 (FP16) | 高 | 基准 | 无 |
| GPTQ 8位 | 中 | 略快 | 极小 |
| GPTQ 4位 | 低 | 快 | 小 |
| GPTQ 4位 + ExLlama | 低 | 非常快 | 小 |
| GPTQ 3位 | 极低 | 快 | 中等 |
| GPTQ 2位 | 极低 | 快 | 较大 |

## 注意事项

1. 首次运行时，脚本会下载校准数据集和模型权重，这可能需要一些时间
2. 量化过程可能需要几分钟到几十分钟，取决于您的硬件和选择的位数
3. 位数越低，内存使用越少，但模型质量可能会有所下降
4. ExLlama 和 Marlin 后端可以显著提高推理速度，但可能需要额外的依赖项
5. 对于 A100 GPU，推荐使用 Marlin 后端；对于其他 GPU，推荐使用 ExLlama v2 后端 