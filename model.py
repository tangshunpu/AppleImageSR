"""PyTorch implementation of Apple's cnn_gp_mlscaler image upscalers.

Input:  float32 [N, 4, H, W], BGRA in [0, 1]
Output: float32 [N, 4, H * scale, W * scale], BGRA in [0, 1]

The original models are CoreML programs (cnn_gp_mlscaler / cnn_gp_mlscaler4x)
shipped in the plaintext Apple Intelligence visual assets.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHT_FILES = {
    2: "cnn_gp_mlscaler_2x.pt",
    4: "cnn_gp_mlscaler_4x.pt",
}


class AppleImageSR(nn.Module):
    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        if scale not in WEIGHT_FILES:
            raise ValueError(f"scale must be one of {sorted(WEIGHT_FILES)}, got {scale}")
        self.scale = scale
        self.scale_factor = float(scale)

        self.bgra2rgb = nn.Conv2d(4, 3, 1, bias=False)
        self.head_0 = nn.Conv2d(3, 128, 3, padding=1)
        self.enc0_0 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc1_0 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.enc2_0 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.enc3_0 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.dec3_0 = nn.Conv2d(512, 512, 1)
        self.dec3_2 = nn.Conv2d(512, 256, 1)
        self.dec2_0 = nn.Conv2d(256, 256, 3, padding=1)
        self.dec2_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec1_0 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec1_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec0_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.dec0_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up_0 = nn.Conv2d(128, 256, 3, padding=1)
        self.up_2 = nn.Conv2d(256 // (scale * scale), 3, 3, padding=1)
        self.identity = nn.Conv2d(3, 3, 1)
        self.rgb2bgra = nn.Conv2d(3, 4, 1)

    @staticmethod
    def _upsample2(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = self.bgra2rgb(x)
        enc_up = F.relu(self.head_0(rgb))

        s0 = F.relu(self.enc0_0(rgb))
        s1 = F.relu(self.enc1_0(s0))
        s2 = F.relu(self.enc2_0(s1))
        y = F.relu(self.enc3_0(s2))

        y = F.relu(self.dec3_0(y))
        y = F.relu(self.dec3_2(y))
        y = self._upsample2(y) + s2

        y = F.relu(self.dec2_0(y))
        y = F.relu(self.dec2_2(y))
        y = self._upsample2(y) + s1

        y = F.relu(self.dec1_0(y))
        y = F.relu(self.dec1_2(y))
        y = self._upsample2(y) + s0

        y = F.relu(self.dec0_0(y))
        y = F.relu(self.dec0_2(y))
        y = y + enc_up

        y = self.up_0(y)
        y = F.pixel_shuffle(y, self.scale)
        y = self.up_2(y)

        skip = F.interpolate(
            self.identity(rgb),
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        y = self.rgb2bgra(y + skip)
        return torch.clamp(y, 0.0, 1.0)


def load_model(
    scale: int = 4,
    weights: str | Path | None = None,
    dtype: torch.dtype = torch.float32,
) -> AppleImageSR:
    path = Path(weights) if weights is not None else WEIGHTS_DIR / WEIGHT_FILES[scale]
    model = AppleImageSR(scale)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(dtype).eval().requires_grad_(False)
