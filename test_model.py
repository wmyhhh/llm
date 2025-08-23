from transformers import AutoModelForCausalLM, AutoTokenizer
from registry import get_quantizer
import torch

def main():
    # 1. 加载模型
    model_name = "facebook/opt-125m"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    # 2. 获取量化器
    QuantizerClass = get_quantizer("bnb")
    quantizer = QuantizerClass(model, quant_type="int8")

    # 3. 执行量化
    quantized_model = quantizer.quantize()

    # 4. 保存模型
    quantizer.save("quantized_models/opt-125m-bnb-int8")

    # 5. 测试推理
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = quantized_model.generate(**inputs, max_new_tokens=20)
    print("Generated text:", tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
