import argparse
from config.parser import load_config
from model.loader import load_model
from quantizer import get_quantizer_class

def run(config_path):
    config = load_config(config_path)
    model, _ = load_model(config["model_path"])

    quant_method = config["quant_method"]
    QuantizerClass = get_quantizer_class(quant_method)
    quantizer = QuantizerClass(config, model)

    quantizer.calibrate(None)
    quantizer.quantize()
    quantizer.save(config["save_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run(args.config)
