"""
Feature Pyramid Vision Transformer (FPViT) for DermaMNIST.

Architecture, adapted for small (28x28) medical images:

  input -> stem conv -> [basic layer 1] -> B1 (28x28x64)  -> ViT head 1 -> a1
                      -> [basic layer 2] -> B2 (14x14x128) -> ViT head 2 -> a2
                      -> [basic layer 3] -> B3 (7x7x256)   -> ViT head 3 -> a3
                      -> [basic layer 4] -> B4 (4x4x512)   -> pool+fc    -> a4

  A = concat(a1, a2, a3, a4) -> fully connected classifier -> logits

Each "basic layer" is two residual BasicBlocks (ResNet-18 stem), matching the
four-stage structure of ResNet-18. Because DermaMNIST images are 28x28 (not
224x224), the usual 7x7/stride-2 stem + max-pool used for ImageNet ResNets is
replaced with a single 3x3/stride-1 stem, which is the standard adaptation for
small-image ResNets (as used for the MedMNIST ResNet-18(28) baseline).

Each shallow ViT head treats every spatial location of its feature map as one
token (1x1 "patch"), linearly projects it to the embedding dimension, adds a
learned position embedding, prepends a learned class token, and runs it
through a small pre-norm Transformer encoder. The class token after the final
layer norm is the activation vector for that scale.

Larger inputs (e.g. 224x224, DermaMNIST-C/E) use the standard ImageNet stem
instead (7x7/stride-2 conv + 3x3/stride-2 max-pool: 224 -> 56 -> 28 -> 14 -> 7),
and each ViT head groups p x p neighbouring locations into one token so that
every head sees the same `token_grid` x `token_grid` tokens (default 14, i.e.
196 tokens: the sequence length of ViT-B/16 at 224). One token per location
would give the first head 56*56 = 3136 tokens, and attention cost grows with
the square of that. With the ImageNet stem the extractor has exactly
torchvision's ResNet-18 layout, so it can start from ImageNet weights
(`load_imagenet_resnet18`).

This module is a from-scratch re-implementation based on the architecture and
equations described in:
  Liu, J., Li, Y., Cao, G., Liu, Y., Cao, W. "Feature Pyramid Vision
  Transformer for MedMNIST Classification Decathlon." IJCNN 2022.
No official code release for this paper was found, so this is an original
implementation of the described design (not a copy of any source).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# ResNet-18 basic block
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Standard ResNet basic block: conv3x3-bn-relu-conv3x3-bn + shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def _make_basic_layer(in_channels: int, out_channels: int, stride: int, num_blocks: int = 2):
    layers = [BasicBlock(in_channels, out_channels, stride)]
    for _ in range(num_blocks - 1):
        layers.append(BasicBlock(out_channels, out_channels, 1))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# Multi-scale ResNet-18 feature extractor (returns B1, B2, B3, B4)
# --------------------------------------------------------------------------
STEMS = ("small", "imagenet")


def default_stem(input_size: int) -> str:
    """The small-image stem up to 64 px (MedMNIST sizes 28/64), ImageNet's above."""
    return "small" if input_size <= 64 else "imagenet"


def default_token_grid(stem: str) -> int | None:
    """None = one token per feature-map location (the paper's 1x1 patches)."""
    return None if stem == "small" else 14


class ResNet18Extractor(nn.Module):
    """ResNet-18 stem + 4 basic layers.

    Feature map sizes with the "small" stem and a 28x28 input:
        B1: (batch,  64, 28, 28)
        B2: (batch, 128, 14, 14)
        B3: (batch, 256,  7,  7)
        B4: (batch, 512,  4,  4)
    and with the "imagenet" stem and a 224x224 input: 56, 28, 14, 7.
    """

    def __init__(self, in_channels: int = 3, stem: str = "small"):
        super().__init__()
        if stem == "small":
            # Small-image stem (no stride-2 7x7 conv / maxpool, unlike ImageNet ResNet).
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        elif stem == "imagenet":
            # torchvision's ResNet-18 stem, module for module (see load_imagenet_resnet18).
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError(f"unknown stem {stem!r}; choose from {STEMS}")
        self.layer1 = _make_basic_layer(64, 64, stride=1)    # -> B1, 28x28
        self.layer2 = _make_basic_layer(64, 128, stride=2)   # -> B2, 14x14
        self.layer3 = _make_basic_layer(128, 256, stride=2)  # -> B3, 7x7
        self.layer4 = _make_basic_layer(256, 512, stride=2)  # -> B4, 4x4

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        return b1, b2, b3, b4


# --------------------------------------------------------------------------
# Pre-norm Transformer encoder layer (matches z' = MSA(LN(z)) + z formulation)
# --------------------------------------------------------------------------
def _make_transformer_encoder(embed_dim: int, depth: int, num_heads: int,
                               mlp_ratio: float, drop_rate: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=drop_rate,
        activation="gelu",
        batch_first=True,
        norm_first=True,  # pre-norm, matches the paper's MSA(LN(.)) formulation
    )
    return nn.TransformerEncoder(layer, num_layers=depth)


# --------------------------------------------------------------------------
# Shallow ViT head operating on one scale of ResNet feature maps
# --------------------------------------------------------------------------
class ShallowViTHead(nn.Module):
    """Turns a (batch, C, H, W) feature map into an activation vector.

    With `patch_size=1` each of the H*W spatial locations is one token,
    linearly projected to `embed_dim` (the paper's design). With
    `patch_size=p > 1` each non-overlapping p x p block is one token,
    projected by a p x p / stride-p convolution as in ViT's patch embedding.
    A learned class token is prepended and a learned position embedding is
    added, following standard ViT practice. The class token's representation
    after the encoder + final LayerNorm is the returned activation vector a_i.
    """

    def __init__(self, in_channels: int, num_patches: int, embed_dim: int = 192,
                 depth: int = 4, num_heads: int = 3, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1, patch_size: int = 1):
        super().__init__()
        # Linear for p=1 keeps the parameter shapes of every existing checkpoint.
        self.proj = (nn.Linear(in_channels, embed_dim) if patch_size == 1 else
                     nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.encoder = _make_transformer_encoder(embed_dim, depth, num_heads, mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        b = feat_map.shape[0]
        if isinstance(self.proj, nn.Conv2d):
            # (b, c, h, w) -> (b, d, h/p, w/p) -> (b, n, d): one token per p x p block
            tokens = self.proj(feat_map).flatten(2).transpose(1, 2)
        else:
            # (b, c, h, w) -> (b, h*w, c): each spatial location is one token
            tokens = self.proj(feat_map.flatten(2).transpose(1, 2))  # (b, n, d)

        cls_tok = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        tokens = self.encoder(tokens)
        cls_out = tokens[:, 0]  # class token output
        return self.norm(cls_out)


# --------------------------------------------------------------------------
# ResNet head for the fourth (deepest) scale, kept as a plain conv-net path
# --------------------------------------------------------------------------
class ResNetHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat_map).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------
# Full FPViT model
# --------------------------------------------------------------------------
class FPViT(nn.Module):
    """Feature Pyramid Vision Transformer.

    Args:
        num_classes: number of output classes (7 for DermaMNIST).
        in_channels: number of input image channels (3 for DermaMNIST).
        input_size: input spatial size (28 for DermaMNIST, 224 for DermaMNIST-C/E).
        stem: "small" (3x3/stride-1, for 28x28) or "imagenet" (7x7/stride-2 +
            max-pool); None picks by input_size (see default_stem).
        token_grid: side of the token grid every ViT head sees; each head
            groups (feature size // token_grid)^2 locations per token. None
            means one token per location (the paper's design, used at 28x28).
        embed_dim: embedding dimension shared by all ViT heads and the
            ResNet head (so their activation vectors can be concatenated).
        depth: number of Transformer layers per ViT head (paper default: 4).
        num_heads: number of attention heads per ViT head.
        mlp_ratio: MLP hidden-dim expansion ratio inside each Transformer layer.
        drop_rate: dropout used in the ViT heads.
        use_resnet_head: whether to keep the 4th (ResNet) head. The paper's
            ablation shows keeping it ("4 heads") outperforms dropping it
            ("3 heads"); set to False to reproduce the 3-head ablation.
        multi_label: if True, the model is meant to be trained with
            BCEWithLogitsLoss (multi-label case). DermaMNIST is single-label
            multi-class, so this should stay False.
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 3, input_size: int = 28,
                 embed_dim: int = 192, depth: int = 4, num_heads: int = 3,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.1,
                 use_resnet_head: bool = True, multi_label: bool = False,
                 stem: str | None = None, token_grid: int | None = None):
        super().__init__()
        self.multi_label = multi_label
        self.use_resnet_head = use_resnet_head
        stem = stem or default_stem(input_size)

        self.backbone = ResNet18Extractor(in_channels=in_channels, stem=stem)

        # Spatial size of B1 after the stem (the imagenet stem halves twice),
        # then B2, B3, B4 at stride 2 each.
        s1 = input_size if stem == "small" else _halve(_halve(input_size))
        s2 = _halve(s1)
        s3 = _halve(s2)

        heads = []
        for channels, size in ((64, s1), (128, s2), (256, s3)):
            patch = 1 if not token_grid else max(1, size // token_grid)
            grid = size // patch
            heads.append(ShallowViTHead(channels, grid * grid, embed_dim, depth, num_heads,
                                        mlp_ratio, drop_rate, patch_size=patch))
        self.vit1, self.vit2, self.vit3 = heads

        num_heads_used = 3
        if use_resnet_head:
            self.resnet_head = ResNetHead(512, embed_dim)
            num_heads_used = 4

        self.classifier = nn.Linear(embed_dim * num_heads_used, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1, b2, b3, b4 = self.backbone(x)
        a1 = self.vit1(b1)
        a2 = self.vit2(b2)
        a3 = self.vit3(b3)

        if self.use_resnet_head:
            a4 = self.resnet_head(b4)
            fused = torch.cat([a1, a2, a3, a4], dim=1)
        else:
            fused = torch.cat([a1, a2, a3], dim=1)

        logits = self.classifier(fused)
        return logits  # raw logits; apply softmax/sigmoid outside for probabilities


def _halve(size: int) -> int:
    """Output size of a stride-2 conv/pool with 'same' padding."""
    return (size + 1) // 2


def build_fpvit(num_classes: int = 7, in_channels: int = 3, input_size: int = 28,
                 embed_dim: int = 192, depth: int = 4, num_heads: int = 3,
                 use_resnet_head: bool = True, multi_label: bool = False,
                 stem: str | None = None, token_grid: int | None = None) -> FPViT:
    """Convenience factory with the paper's default hyperparameters
    (4 Transformer layers per ViT head, 4 heads including the ResNet head)."""
    return FPViT(
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        use_resnet_head=use_resnet_head,
        multi_label=multi_label,
        stem=stem,
        token_grid=token_grid,
    )


def load_imagenet_resnet18(model: FPViT) -> list[str]:
    """Copies torchvision's ImageNet ResNet-18 weights into the extractor.

    Needs the "imagenet" stem: with it, stem.0/stem.1 are torchvision's
    conv1/bn1 and layer1..layer4 have the same module names, so the mapping is
    one-to-one. The ViT heads, the ResNet head and the classifier stay randomly
    initialised. ImageNet only: no dermoscopy data, so no leakage into
    DermaMNIST's validation or test split. Returns the loaded parameter names.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    stem = model.backbone.stem
    if len(stem) != 4 or stem[0].kernel_size != (7, 7):
        raise ValueError("ImageNet weights need the 'imagenet' stem (7x7 conv + max-pool)")
    src = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    rename = {"conv1.": "stem.0.", "bn1.": "stem.1."}
    mapped = {}
    for key, value in src.items():
        if key.startswith("fc."):
            continue
        for old, new in rename.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        mapped[key] = value
    model.backbone.load_state_dict(mapped, strict=True)
    return sorted(mapped)


if __name__ == "__main__":
    # Quick shape sanity check on random data, at both supported resolutions.
    for size in (28, 224):
        model = build_fpvit(num_classes=7, in_channels=3, input_size=size,
                            token_grid=default_token_grid(default_stem(size)))
        out = model(torch.randn(2, 3, size, size))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{size}x{size}: output {tuple(out.shape)}, {n_params:,} parameters")
