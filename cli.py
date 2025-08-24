import argparse
import os
import shutil
import torch
from transformers import AutoTokenizer

#from .registry import get_quantizer
#from .methods import GPTQQuantizer, BnBQuantizer, AQLMQuantizer  # 确保量化方法被注册
from .methods import *  # 确保量化方法被注册

# copy tokenizer files
def copy_tokenizer(src, dst):
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    for f in tokenizer_files:
        src_file = os.path.join(src, f)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst)
            print(f"[Info] 已复制 {f} 到 {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Quantization CLI - supports BnB and GPTQ methods"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace Hub 模型名称 (可选)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="本地模型路径 (可选)"
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=["4bit", "8bit", "2bit"],
        default="4bit",
        help="量化类型: 2bit, 4bit 或 8bit"
    )
    parser.add_argument(
        "--bnb_dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="4bit 量化计算类型 (仅 4bit 有效)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="模型放置设备: auto / cuda / cpu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="量化模型保存路径"
    )
    # 是否保存原 tokenizer
    parser.add_argument(
        "--save_tokenizer",
        action="store_true",
        help="是否将原模型的 tokenizer 一起复制到保存目录"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["bnb", "gptq", "aqlm", "awq"],
        default="bnb",
        help="选择量化方法: bnb 或 gptq"
    )

    args = parser.parse_args()

    # 判断用户到底是给了 name 还是 path
    if args.model_path:
        model_name_or_path = args.model_path
    elif args.model_name:
        model_name_or_path = args.model_name
    else:
        raise ValueError("请提供 --model_name 或 --model_path 中至少一个")

    # 将 dtype 字符串转换为 torch 类型
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    bnb_dtype = dtype_map[args.bnb_dtype]

    # 获取量化器
    # 获取量化器
    #QuantizerClass = get_quantizer(args.method)

    #print("model_name_or_path:", model_name_or_path, type(model_name_or_path))

    # 初始化量化器
    if args.method == "bnb":
        quantizer = BnBQuantizer(
            model=model_name_or_path,
            quant_type=args.quant_type,
            bnb_4bit_compute_dtype=bnb_dtype,
            device_map=args.device,
            save_dir=args.save_dir  # 如果用户没传，这里为 None，会使用类内默认 "bnb"
        )
    elif args.method == "gptq":
        quantizer = GPTQQuantizer(
            model=model_name_or_path,
            quant_type=args.quant_type,
            device_map=args.device,
            save_dir=args.save_dir  # 默认 "gptq"
        )
    elif args.method == "aqlm":
        quantizer = AQLMQuantizer(
            model=model_name_or_path,
            quant_type=args.quant_type,
            device_map=args.device,
            save_dir=args.save_dir  # 默认 "aqlm"
        )
    elif args.method == "awq":
        quantizer = AWQQuantizer(
            model=model_name_or_path,
            quant_type=args.quant_type,
            device_map=args.device
        )
    else:
        raise ValueError(f"Unsupported quantization method: {args.method}")

    print(f"开始量化模型 {model_name_or_path} ...")
    quantized_model = quantizer.quantize()

    # 保存路径使用量化器内部的 save_dir 默认值
    save_path = os.path.abspath(quantizer.save_dir)
    quantizer.save(save_path)
    print(f"量化模型已保存到: {save_path}")


    # 可选：复制 tokenizer
    if args.save_tokenizer:
        copy_tokenizer(model_name_or_path, save_path)
        print("[Info] tokenizer 文件已保存到量化目录")


if __name__ == "__main__":
    main()
