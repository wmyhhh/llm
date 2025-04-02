import torch
print(torch.cuda.is_available())  # 如果是 False，说明 PyTorch 没有检测到 GPU
print(torch.version.cuda)  # 检查 PyTorch 关联的 CUDA 版本
print(torch.backends.cudnn.version())  # 检查 cuDNN 版本
