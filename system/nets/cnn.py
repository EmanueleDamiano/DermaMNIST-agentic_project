"""
Three compact CNN baselines for 28x28 DermaMNIST, alongside FPViT.

The same three families the AgenticDerma proposal compares (ResNet-18,
EfficientNet-B0, ConvNeXt-Tiny), rebuilt for 28x28 inputs: the ImageNet stems
(a stride-2 7x7 conv + max-pool, or a 4x4 patchify) would shrink a 28x28 image
to 7x7 before the first block and leave nothing for the later stages. Every
model here keeps full resolution in its first stage and downsamples
28 -> 14 -> 7 -> 4, the same pyramid FPViT's extractor uses, so the four
architectures differ in their blocks, not in how much spatial detail they are
allowed to see.

All return raw logits of shape (batch, num_classes).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .model import ResNet18Extractor


# ---------------------------------------------------------------------------
# ResNet-18: FPViT's own extractor with a plain classification head
# ---------------------------------------------------------------------------

class ResNet18Small(nn.Module):
    """ResNet-18 (small-image stem) + global average pooling + linear.

    It is exactly FPViT's backbone without the ViT heads, which makes it the
    natural ablation: any gap between the two is what the heads contribute.
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        self.backbone = ResNet18Extractor(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b4 = self.backbone(x)[-1]
        return self.fc(self.dropout(torch.flatten(F.adaptive_avg_pool2d(b4, 1), 1)))


# ---------------------------------------------------------------------------
# EfficientNet-B0: MBConv + squeeze-excitation, B0 widths and depths
# ---------------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, squeezed: int):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, squeezed, 1)
        self.fc2 = nn.Conv2d(squeezed, channels, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class MBConv(nn.Module):
    def __init__(self, cin: int, cout: int, expand: int, kernel: int, stride: int):
        super().__init__()
        hidden = cin * expand
        self.use_residual = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU()]
        layers += [nn.Conv2d(hidden, hidden, kernel, stride, kernel // 2, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), nn.SiLU(),
                   SqueezeExcite(hidden, max(1, cin // 4)),
                   nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class EfficientNetB0Small(nn.Module):
    # (expand, out_channels, repeats, first_stride, kernel) - the B0 table,
    # with the ImageNet stem stride and the first two stride-2 stages turned
    # into stride 1 so a 28x28 input ends at 4x4, not 1x1.
    STAGES = [(1, 16, 1, 1, 3), (6, 24, 2, 1, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3),
              (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]

    def __init__(self, num_classes: int = 7, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.SiLU())
        blocks, cin = [], 32
        for expand, cout, repeats, stride, kernel in self.STAGES:
            for i in range(repeats):
                blocks.append(MBConv(cin, cout, expand, kernel, stride if i == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(cin, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.SiLU())
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.head(self.blocks(self.stem(x)))
        return self.fc(self.dropout(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)))


# ---------------------------------------------------------------------------
# ConvNeXt-Tiny: T widths and depths, small-image stem
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for NCHW tensors."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 3, 1, 2)


class ConvNeXtTinySmall(nn.Module):
    DIMS = (96, 192, 384, 768)
    DEPTHS = (3, 3, 9, 3)

    def __init__(self, num_classes: int = 7, in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        # 3x3 stride-1 stem instead of the 4x4 patchify: 28x28 stays 28x28.
        self.stem = nn.Sequential(nn.Conv2d(in_channels, self.DIMS[0], 3, 1, 1), LayerNorm2d(self.DIMS[0]))
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(self.DIMS, self.DEPTHS)):
            layers = []
            if i > 0:   # 28 -> 14 -> 7 -> 4
                layers += [LayerNorm2d(self.DIMS[i - 1]),
                           nn.Conv2d(self.DIMS[i - 1], dim, 3, stride=2, padding=1)]
            layers += [ConvNeXtBlock(dim) for _ in range(depth)]
            self.stages.append(nn.Sequential(*layers))
        self.norm = nn.LayerNorm(self.DIMS[-1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.DIMS[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x.mean(dim=(2, 3)))
        return self.fc(self.dropout(x))
