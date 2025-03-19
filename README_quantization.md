# Qwen2.5-3B 量化版本运行示例

这个仓库包含了使用 bitsandbytes 量化运行 Qwen/Qwen2.5-3B 大语言模型的示例代码。

## 量化简介

量化是一种通过降低模型权重精度来减少内存使用和加速推理的技术。本示例使用 bitsandbytes 库实现了以下量化方法：

- **8位量化 (INT8)**: 将模型权重从 FP16/FP32 转换为 INT8，可减少约 50% 的内存使用
- **4位量化 (INT4)**: 将模型权重从 FP16/FP32 转换为 INT4，可减少约 75% 的内存使用
  - **FP4**: 标准的 4 位浮点量化
  - **NF4**: 专为神经网络设计的 4 位量化格式，通常有更好的性能
  - **嵌套量化**: 进一步压缩量化常数，额外节省约 0.4 位/参数

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU (推荐至少 8GB 显存)
- 对于 8 位量化，CUDA 11.0+ 和 Compute Capability 7.0+
- 对于 4 位量化，CUDA 11.8+ 和 Compute Capability 7.5+

## 安装依赖

```bash
pip install -r requirements_quantization.txt
```

## 运行模型

### 基本用法

```bash
# 使用默认的 8 位量化
python bitsandbytes.py
```

### 高级用法

```bash
# 不使用量化
python bitsandbytes.py --quantization none

# 使用 8 位量化
python bitsandbytes.py --quantization 8bit

# 使用 4 位量化 (FP4)
python bitsandbytes.py --quantization 4bit

# 使用 4 位量化 (NF4)
python bitsandbytes.py --quantization 4bit --nf4

# 使用 4 位量化 + 嵌套量化
python bitsandbytes.py --quantization 4bit --nf4 --double_quant

# 指定计算数据类型
python bitsandbytes.py --quantization 4bit --compute_dtype float16
```

## 参数说明

- `--quantization`: 量化方法 (`none`, `8bit`, `4bit`)
- `--nf4`: 使用 NF4 数据类型进行 4 位量化
- `--double_quant`: 使用嵌套量化进一步减少内存使用
- `--compute_dtype`: 计算数据类型 (`float32`, `float16`, `bfloat16`)

## 性能比较

| 量化方法 | 内存使用 | 推理速度 | 质量影响 |
|---------|---------|---------|---------|
| 无量化 (FP16) | 高 | 基准 | 无 |
| 8位量化 (INT8) | 中 | 略慢 | 极小 |
| 4位量化 (FP4) | 低 | 慢 | 小 |
| 4位量化 (NF4) | 低 | 慢 | 很小 |
| 4位量化 + 嵌套量化 | 极低 | 慢 | 小 |

## 注意事项

1. 量化会对模型质量产生一定影响，通常 8 位量化的影响很小，4 位量化的影响较大
2. 嵌套量化可以进一步减少内存使用，但可能会略微降低模型质量
3. 首次运行时，脚本会自动从 Hugging Face 下载模型权重，这可能需要一些时间
4. 对于较小的模型（如 Qwen2.5-3B），量化的内存节省效果可能不如对大模型明显 