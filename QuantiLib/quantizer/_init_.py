
from .gptq import GPTQQuant
from .bitsandbytes import BitsAndBytesQuant


# 注册表
QUANTIZER_REGISTRY = {
   
    "GPTQ": GPTQQuant,
    "BitsAndBytes": BitsAndBytesQuant,
   
}

def get_quantizer_class(name: str):
    if name not in QUANTIZER_REGISTRY:
        raise ValueError(f"Unsupported quant_method: {name}")
    return QUANTIZER_REGISTRY[name]
