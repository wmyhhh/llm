from .bnb import BnBQuantizer
from .gptq import GPTQQuantizer
from .aqlm import AQLMQuantizer
from .awq import AWQQuantizer

QUANTIZER_REGISTRY = {
    "bnb": BnBQuantizer,
    "gptq": GPTQQuantizer,
    "aqlm": AQLMQuantizer,
    "awq": AWQQuantizer
}

bnb_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['4bit', '8bit'],  # "4bit" or "8bit"
    "bnb_4bit_compute_dtype": ['float32', 'bfloat16'],  # Only for 4bit
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],
    "bnb_4bit_quant_type": ["nf4", "fp4"], 
    "bnb_4bit_use_double_quant": [True, False] # Only for 4bit
}

gptq_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['2bit', '3bit', '4bit'],  # "2bit", "3bit", or "4bit"
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],
    "batch_size": [int],  # 校准时的 batch size
    "calib_dataset": [str, list],  # 校准数据集
    "gptq_group_size": [int],  # GPTQ 的 group size
}

awq_args = {
    "model_path": [str],
    "model_name": [str],
    "quant_type": ['2bit', '3bit', '4bit'],  # "2bit", "3bit", or "4bit"
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],  
    "group_size": [int],  # AWQ 的 group size
}

aqlm_args = {
    "model_path": [str],
    "model_name": [str],
    "device_map": ["auto", "cuda", "cpu"],
    "save_tokenizer": [True, False],
    "save_dir": [str],  
}


METHOD_ARGS = {
    "gptq": gptq_args,
    "bnb": bnb_args,
    #"aqlm": 
    "awq": awq_args
}

