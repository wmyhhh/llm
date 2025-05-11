from .base import BaseQuantizer

class GPTQQuant(BaseQuantizer):
    def calibrate(self, data_loader):
        print("Running GPTQ calibration...")

    def quantize(self):
        print("Applying GPTQ quantization...")

    def save(self, path):
        print(f"Saving GPTQ quantized model to {path}")
