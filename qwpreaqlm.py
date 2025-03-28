# 创建一个名为 create_aqlm_model.py 的文件
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AqlmConfig
from pathlib import Path

# 设置输出目录
output_dir = Path("./qwen_aqlm_model")
output_dir.mkdir(exist_ok=True, parents=True)

# 加载原始模型
model_name = "Qwen/Qwen2.5-3B"
print(f"加载原始模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 创建 AQLM 配置
aqlm_config = AqlmConfig(
    nbits=1,  # 码本数量，通常为1
    bits=16,  # 码本大小，通常为16
    group_size=16  # 量化分组大小
)

# 加载并量化模型
print("开始 AQLM 量化...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=aqlm_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 保存量化后的模型
print(f"保存 AQLM 量化模型到: {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("完成!")