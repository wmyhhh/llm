# AQLM 量化指南

## 简介

AQLM (Additive Quantization of Language Models) 是一种高效的模型量化方法，它通过将多个权重一起量化并利用它们之间的相互依赖关系来实现更好的压缩效果。AQLM 将 8-16 个权重表示为多个向量码的和，从而在保持模型性能的同时大幅减小模型体积。

本指南将帮助您使用 `qwaqlm.py` 脚本对大型语言模型（如 Qwen 系列）进行 AQLM 量化。

## 安装依赖

在开始之前，请确保安装以下依赖：

```bash
# 安装 AQLM 库（同时支持 GPU 和 CPU 推理）
pip install aqlm[gpu,cpu]

# 安装其他依赖
pip install torch transformers
```

**注意**：AQLM 需要 Python 3.10 或更高版本。

## 使用方法

### 基本用法

最简单的用法是使用默认参数进行量化：

```bash
python qwaqlm.py
```

这将使用默认设置（1x16 配置）对 Qwen/Qwen2.5-3B 模型进行 AQLM 量化，并将结果保存到 `./qwen_aqlm` 目录。

### 高级选项

`qwaqlm.py` 脚本提供了多种参数来自定义量化过程：

```bash
python qwaqlm.py \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --output_dir ./qwen_aqlm_custom \
  --num_codebooks 2 \
  --codebook_size 8 \
  --use_cuda \
  --device cuda:0 \
  --save_results
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 模型路径或 Hugging Face 模型标识符 | Qwen/Qwen2.5-3B |
| `--output_dir` | 保存量化后模型的目录 | ./qwen_aqlm |
| `--num_codebooks` | AQLM 编码本数量（1, 2, 4, 8） | 1 |
| `--codebook_size` | 编码本大小（8 或 16 位） | 16 |
| `--use_triton` | 是否使用 Triton 后端（支持任意编码本配置，但速度较慢） | False |
| `--use_cuda` | 是否使用 CUDA 后端（仅支持 1x16 和 2x8 配置，但速度更快） | False |
| `--use_numba` | 是否使用 Numba 后端（仅支持 Kx8 配置，适用于 CPU 推理） | False |
| `--device` | 加载模型的设备 | cuda:0 |
| `--save_results` | 是否保存性能测试结果 | False |

## AQLM 配置说明

AQLM 量化设置主要根据编码本数量和编码本大小（位）来区分。以下是常见配置及其支持的后端：

| 配置 | 编码本数量 | 编码本大小 | 精度 | 加速比 | 支持的后端 | 适用场景 |
|------|-----------|-----------|------|--------|------------|---------|
| 1x16 | 1 | 16 | 最佳 | 最高 ~1.3x | CUDA, Triton | 需要高精度的场景 |
| 2x8 | 2 | 8 | 良好 | 最高 ~3.0x | CUDA, Triton | 平衡精度和速度 |
| 4x8 | 4 | 8 | 良好 | 最高 ~4.0x | Triton, Numba | 更注重速度 |
| 8x8 | 8 | 8 | 良好 | 最高 ~4.0x | Triton, Numba | 更注重速度 |

## 后端选择指南

如果您没有指定后端，脚本会根据您的配置自动选择最合适的后端：

1. **CUDA 后端**：
   - 仅支持 1x16 和 2x8 配置
   - 提供最快的 GPU 推理速度
   - 适用于需要高性能的场景

2. **Triton 后端**：
   - 支持任意编码本配置
   - GPU 推理速度适中
   - 适用于需要灵活配置的场景

3. **Numba 后端**：
   - 仅支持 Kx8 配置（K 为编码本数量）
   - 提供 CPU 推理支持
   - 适用于没有 GPU 的环境

## 高级用法示例

### 使用 2x8 配置和 CUDA 后端

这是一个平衡精度和速度的常用配置：

```bash
python qwaqlm.py \
  --num_codebooks 2 \
  --codebook_size 8 \
  --use_cuda
```

### 使用 4x8 配置和 Triton 后端

这个配置提供更好的压缩率：

```bash
python qwaqlm.py \
  --num_codebooks 4 \
  --codebook_size 8 \
  --use_triton
```

### 在 CPU 上使用 Numba 后端

如果您没有 GPU 或想在 CPU 上运行：

```bash
python qwaqlm.py \
  --num_codebooks 4 \
  --codebook_size 8 \
  --use_numba \
  --device cpu
```

### 保存性能测试结果

如果您想保存性能测试结果以便后续分析：

```bash
python qwaqlm.py --save_results
```

这将在 `performance_results` 目录中创建一个包含详细测试数据的 JSON 文件。

## 脚本功能

`qwaqlm.py` 脚本执行以下操作：

1. **加载和量化模型**：使用 AQLM 方法加载并量化指定的模型
2. **保存量化后的模型**：将量化后的模型和分词器保存到指定目录
3. **性能评估**：对模型进行性能测试，连续询问 5 次相同的问题
4. **交互式对话**：测试完成后，允许用户与模型进行交互式对话

## 使用量化后的模型

量化完成后，您可以使用以下代码加载和使用量化后的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载量化后的模型和分词器
model_path = "./qwen_aqlm"
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

## AQLM 与其他量化方法的比较

| 量化方法 | 支持的位宽 | 内存节省 | 性能保持 | 推理速度 | 适用场景 |
|---------|-----------|---------|---------|---------|---------|
| AQLM    | 可变 (1x16, 2x8 等) | ~4-8倍 | 极好    | 快      | 需要高性能的低位量化 |
| AWQ     | 主要是 4 位 | ~4倍    | 极好    | 快      | 需要高性能的 4 位量化 |
| GPTQ    | 2-8 位    | 2-8倍   | 良好    | 快      | 需要更灵活位宽选择 |
| BitsAndBytes | 8/4 位 | 2-4倍  | 良好    | 中      | 简单实现，易于使用 |

## 常见问题

### 1. AQLM 与其他量化方法的区别是什么？

AQLM 是一种加性量化方法，它将多个权重一起量化并表示为多个向量码的和。与其他方法相比，AQLM 在相同压缩率下通常能保持更好的模型性能，特别是在低位量化（如 2 位）时。

### 2. 如何选择合适的 AQLM 配置？

- 如果您注重精度：选择 1x16 配置
- 如果您需要平衡精度和速度：选择 2x8 配置
- 如果您更注重速度和内存节省：选择 4x8 或 8x8 配置

### 3. 为什么我的 AQLM 量化模型加载失败？

确保您已正确安装 `aqlm` 库，并且使用的是 Python 3.10 或更高版本。某些配置可能需要特定的后端支持，请检查您的配置是否与所选后端兼容。

### 4. 如何在 CPU 上使用 AQLM 量化模型？

使用 Numba 后端和 Kx8 配置，并将设备设置为 "cpu"：

```bash
python qwaqlm.py --num_codebooks 4 --codebook_size 8 --use_numba --device cpu
```

### 5. 如何退出交互式对话模式？

在对话模式中输入 'exit' 即可退出程序。

## 结论

AQLM 量化是一种强大的模型压缩方法，可以显著减少模型大小和内存占用，同时保持良好的性能。通过本指南中的 `qwaqlm.py` 脚本，您可以轻松地对 Qwen 系列模型进行 AQLM 量化，并根据自己的需求调整各种参数。

希望本指南对您有所帮助！如有任何问题，请随时提问。 