import torch

from model import load_model

for scale in (2, 4):
    model = load_model(scale)
    x = torch.rand(1, 4, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"{scale}x: {tuple(x.shape)} -> {tuple(y.shape)}")
