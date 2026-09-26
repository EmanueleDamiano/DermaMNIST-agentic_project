"""
One constructor for every architecture train.py can train and the prediction
agent can load.

A checkpoint's `config` is train.py's `vars(args)`. Checkpoints written before
`--arch` existed have no "arch" key and are FPViT - `build_from_config` keeps
reading them exactly as before.
"""

from __future__ import annotations

from torch import nn

from .cnn import ConvNeXtTinySmall, EfficientNetB0Small, ResNet18Small
from .model import build_fpvit

DEFAULT_ARCH = "fpvit"

ARCHITECTURES = {
    "fpvit": "Feature Pyramid ViT: ResNet-18 extractor + 3 shallow ViT heads + ResNet head (IJCNN 2022)",
    "resnet18": "ResNet-18 with a small-image stem, global pooling + linear head",
    "efficientnet_b0": "EfficientNet-B0 (MBConv + squeeze-excitation), stride schedule adapted to 28x28",
    "convnext_tiny": "ConvNeXt-Tiny (7x7 depthwise, LayerNorm, GELU), small-image stem",
}


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 7, in_channels: int = 3,
                embed_dim: int = 192, depth: int = 4, num_heads: int = 3,
                use_resnet_head: bool = True) -> nn.Module:
    """The FPViT-specific arguments are ignored by the CNNs."""
    if arch == "fpvit":
        return build_fpvit(num_classes=num_classes, in_channels=in_channels, input_size=28,
                           embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                           use_resnet_head=use_resnet_head)
    if arch == "resnet18":
        return ResNet18Small(num_classes, in_channels)
    if arch == "efficientnet_b0":
        return EfficientNetB0Small(num_classes, in_channels)
    if arch == "convnext_tiny":
        return ConvNeXtTinySmall(num_classes, in_channels)
    raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")


def build_from_config(cfg: dict, num_classes: int = 7, in_channels: int = 3) -> nn.Module:
    """Rebuild the network a checkpoint was trained with, from its stored config."""
    return build_model(
        arch=cfg.get("arch") or DEFAULT_ARCH,
        num_classes=num_classes, in_channels=in_channels,
        embed_dim=cfg.get("embed_dim", 192), depth=cfg.get("depth", 4),
        num_heads=cfg.get("num_heads", 3),
        use_resnet_head=not cfg.get("no_resnet_head", False),
    )
