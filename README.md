# AppleImageSR

PyTorch ports of Apple's `cnn_gp_mlscaler` (2x) and `cnn_gp_mlscaler4x` (4x) image
upscalers, converted from the CoreML programs shipped in the plaintext Apple
Intelligence visual assets (Image Playground upscaler).

## Structure

```
AppleImageSR/
├── model.py          # model definition + load_model(scale)
├── weights/
│   ├── cnn_gp_mlscaler_2x.pt
│   └── cnn_gp_mlscaler_4x.pt
├── example.py        # minimal inference example
└── requirements.txt
```

## Usage

```python
import torch
from model import load_model

model = load_model(4)                       # 2 or 4
x = torch.rand(1, 4, 256, 256)              # NCHW BGRA, values in [0, 1]
with torch.no_grad():
    y = model(x)                            # (1, 4, 1024, 1024)
```

```bash
pip install -r requirements.txt
python example.py
```

Input is BGRA, output is BGRA in `[0, 1]`. Spatial sizes are dynamic.

## Model

- ~3.57M parameters, fp32 weights
- Encoder: 3→64→128→256→512 with stride-2 3×3 convolutions
- Decoder: 2× bilinear upsampling with skip connections, 3×3 convolutions
- Output head: 3×3 conv → `pixel_shuffle(scale)` → 3×3 conv, plus an upsampled
  identity skip from the RGB input
- The 2x and 4x checkpoints differ in the `up_2` shape and the identity-skip
  weights (identity in the 2x model, learned 3×3 matrix in the 4x model)

## Validation against CoreML

CPU-only comparisons with the original `.mlmodelc` on random inputs:

| Model | Input | Output | PSNR | Mean abs error | Max abs error |
| --- | --- | --- | --- | --- | --- |
| 2x | 256×256 | 512×512 | 60.15 dB | 5.74e-4 | 7.56e-3 |
| 2x | 768×768 | 1536×1536 | 60.11 dB | 5.76e-4 | 8.60e-3 |
| 4x | 256×256 | 1024×1024 | 54.64 dB | 9.76e-4 | 2.00e-2 |
| 4x | 512×512 | 2048×2048 | 54.62 dB | 9.77e-4 | 2.26e-2 |
| 4x | 1024×512 | 4096×2048 | 54.60 dB | 9.80e-4 | 2.30e-2 |

The alpha channel matches exactly. Residuals come from fp16 CoreML storage and
accumulation order, not from model topology.

## Provenance

Weights were extracted from Apple's plaintext CoreML assets and converted to
PyTorch state dicts. They remain Apple's property and are provided for private
research and interoperability testing only. Do not redistribute.
