"""
One constructor for every architecture train.py can train and the prediction
agent can load.

A checkpoint's `config` is train.py's `vars(args)`. Checkpoints written before
`--arch` existed have no "arch" key and are FPViT, and those written before
`--img-size` existed have no "img_size" / "stem" / "token_grid" keys and are
28x28 - `build_from_config` keeps reading both exactly as before.
"""

from __future__ import annotations

from torch import nn

from .cnn import ConvNeXtTinySmall, EfficientNetB0Small, ResNet18Small
from .model import build_fpvit, default_stem, default_token_grid

DEFAULT_ARCH = "fpvit"

ARCHITECTURES = {
    "fpvit": "Feature Pyramid ViT: ResNet-18 extractor + 3 shallow ViT heads + ResNet head (IJCNN 2022)",
    "resnet18": "ResNet-18 with a small-image stem, global pooling + linear head",
    "efficientnet_b0": "EfficientNet-B0 (MBConv + squeeze-excitation), stride schedule adapted to 28x28",
    "convnext_tiny": "ConvNeXt-Tiny (7x7 depthwise, LayerNorm, GELU), small-image stem",
}


# The CNNs keep a stride-1 small-image stem: at 224x224 they would run their
# whole first stage at full resolution, so they are trained at 28 only.
CNN_IMG_SIZES = (28,)


def resolve_fpvit_shape(img_size: int = 28, stem: str | None = None,
                        token_grid: int | None = None) -> tuple[str, int | None]:
    """Fills in the stem and token grid a given input size defaults to.

    train.py stores the resolved values in the checkpoint config, so a model
    is always rebuilt with the shape it was trained with, even if these
    defaults change later. token_grid 0 is the CLI's way of saying "one token
    per location" and is returned as None.
    """
    stem = stem or default_stem(img_size)
    if token_grid is None:
        token_grid = default_token_grid(stem)
    return stem, (token_grid or None)


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 7, in_channels: int = 3,
                embed_dim: int = 192, depth: int = 4, num_heads: int = 3,
                use_resnet_head: bool = True, img_size: int = 28,
                stem: str | None = None, token_grid: int | None = None) -> nn.Module:
    """The FPViT-specific arguments are ignored by the CNNs."""
    if arch == "fpvit":
        stem, token_grid = resolve_fpvit_shape(img_size, stem, token_grid)
        return build_fpvit(num_classes=num_classes, in_channels=in_channels, input_size=img_size,
                           embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                           use_resnet_head=use_resnet_head, stem=stem, token_grid=token_grid)
    if img_size not in CNN_IMG_SIZES:
        raise ValueError(f"{arch} is built for {CNN_IMG_SIZES} px inputs, not {img_size}; "
                         "use --arch fpvit at higher resolution")
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
        img_size=cfg.get("img_size", 28),
        stem=cfg.get("stem"),
        token_grid=cfg.get("token_grid"),
    )
