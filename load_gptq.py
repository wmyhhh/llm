import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型和 tokenizer 路径
MODEL_DIR = "/home/lxrobotlab-4090-a/wmy/quantized_model"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 加载量化后的模型
# device_map='auto' 会自动选择 GPU，如果没有 GPU 可以改为 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map='auto',  # 或者 device_map={'': 'cuda:0'} 指定 GPU
    torch_dtype=torch.float16,  # GPTQ 量化模型通常用 float16
)

# 设置模型为 eval 模式
model.eval()

# Interactive chat
print("=== Interactive Chat Mode ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Encode user input
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("Bot:", response)

