"""
Stage 2 — Baseline U-Net (single Nucleus head)
================================================

A conventional encoder-decoder U-Net with skip connections, one output
head predicting the binary nucleus probability map. This is the control
model: everything in Stage 3 (boundary head) and Stage 4 (distance head)
gets compared back against this baseline.

Kept deliberately plain -- no boundary head, no distance head yet. Those
get added in later stages so we can isolate exactly how much each one
contributes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convs, each followed by BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Maxpool then ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample, concat with skip connection, then ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if spatial dims mismatch by 1px due to odd input sizes
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    """
    Standard U-Net. `base_ch` controls capacity (default 32 -> channels
    32/64/128/256/512, matching the classic 64-based U-Net scaled down for
    faster iteration; bump to 64 once the pipeline is validated).

    Single output head: nucleus binary probability map, 1 channel, raw
    logits (sigmoid applied in the loss / at inference, not in the model).
    """

    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)

        self.nucleus_head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        nucleus_logits = self.nucleus_head(x)
        return {"nucleus": nucleus_logits}


if __name__ == "__main__":
    # smoke test: forward pass on a random batch, check shapes
    model = UNetBaseline(in_channels=3, base_ch=32)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("nucleus_logits shape:", out["nucleus"].shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")
    assert out["nucleus"].shape == (2, 1, 256, 256), "shape mismatch!"
    print("OK: baseline U-Net forward pass shape check passed.")