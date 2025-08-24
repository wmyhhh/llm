import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Path to your quantized model
model_path = "/home/lxrobotlab-4090-a/wmy/bnb"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Load quantized model
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)
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

