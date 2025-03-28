# EETQ量化方法

EETQ是一种高效的权重量化工具，专为NVIDIA GPU上的大型语言模型优化设计。它提供了高性能的INT8量化实现，且无需校准数据集。

## 特点与优势

- **无需校准数据集**：EETQ不需要预先准备校准数据集，简化了量化流程
- **高性能内核**：使用来自FasterTransformer和TensorRT-LLM的高性能GEMM和GEMV内核
- **精度损失极小**：通过每通道量化方法，保持模型精度几乎不受影响
- **即时量化**：无需预先量化模型，可直接加载原始模型并进行量化
- **兼容性强**：适用于CUDA能力为7.0至8.9的NVIDIA GPU

## 安装方法

可以通过以下两种方式安装EETQ：

### 方法一：直接安装预编译包

```bash
pip install --no-cache-dir https://github.com/NetEase-FuXi/EETQ/releases/download/v1.0.0/EETQ-1.0.0+cu121+torch2.1.2-cp310-cp310-linux_x86_64.whl
```

### 方法二：从源码安装

```bash
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

## 脚本使用说明

本仓库提供了`qweetq.py`脚本，用于对Qwen2.5-3B模型使用EETQ进行INT8量化。

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.36.0+
- EETQ 库
- CUDA 11.8+

### 运行方法

1. 确保已安装EETQ库
2. 运行以下命令：

```bash
python qweetq.py --save_results
```

### 参数说明

- `--bits`: 量化位数，目前EETQ仅支持int8
- `--save_results`: 是否保存性能测试结果

### 脚本功能

`qweetq.py`脚本提供以下功能：

1. 使用EETQ将Qwen2.5-3B模型量化为INT8精度
2. 比较量化前后的内存使用和推理速度
3. 提供交互式对话接口测试量化模型
4. 可选保存量化后的模型以供后续使用

### 保存和重用量化模型

脚本允许保存量化模型，您可以在测试完成后选择是否保存。保存后的模型可以通过以下代码加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("qwen2.5-3b-eetq-int8", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("qwen2.5-3b-eetq-int8")
```

## 性能对比

EETQ量化通常可以提供以下性能改进：

- 内存使用减少约50%（相比FP16模型）
- 保持接近FP16模型的推理精度
- 可能带来轻微的推理速度提升

## 已知限制

- 当前版本仅支持INT8量化
- 需要CUDA能力在7.0和8.9之间的NVIDIA GPU
- 与某些版本的PyTorch和CUDA可能存在兼容性问题

## 参考资料

- [EETQ GitHub 仓库](https://github.com/NetEase-FuXi/EETQ)
- [Hugging Face Transformers EETQ 文档](https://huggingface.co/docs/transformers/quantization/eetq) 