import torch
import os
from .methods.gptq import * # 确保量化方法被注册

def main():
    model_name = "Qwen/Qwen3-1.7B"  # HuggingFace model name
    save_dir = "./qwen3_1.7B_gptq"

    # Get the BitsAndBytes quantizer
    QuantizerClass = GPTQQuantizer
    quantizer = QuantizerClass(
        model=model_name,
        quant_type="4bit",
        device_map="auto"  # let HuggingFace decide device placement
    )

    print(f"Start quantizing {model_name} to 4-bit ...")
    quantized_model = quantizer.quantize()

    # Save quantized model
    os.makedirs(save_dir, exist_ok=True)
    quantizer.save(save_dir)
    print(f"Quantized model saved at {save_dir}")

if __name__ == "__main__":
    main()
