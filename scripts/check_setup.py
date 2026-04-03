import torch
from PIL import Image
import diffusers
import transformers
import peft


print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"Diffusers: {diffusers.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")

# Quick MPS tensor test
device = torch.device("mps")
x = torch.randn(3, 3).to(device)
print(f"MPS tensor test: {x.shape} on {x.device} \u2713")
