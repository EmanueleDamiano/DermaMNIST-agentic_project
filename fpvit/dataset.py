"""
DermaMNIST data loading for the FPViT baseline.

Uses the official `medmnist` package (MedMNIST v2), which ships the fixed
train/val/test split used for all MedMNIST Classification Decathlon
benchmarks (train 7007 / val 1003 / test 2005).

Augmentation lives in `fpvit/augment.py` and is described by an
`AugmentationConfig`, which this module simply applies to the train split.
Val and test are never augmented. Pass `aug=None` to train without any
augmentation.

Note on the AgenticDerma project proposal: WP2 (Data) calls for an
integrity audit of DermaMNIST for duplicates / cross-split leakage before
training (see Abhishek, Jain & Hamarneh, cited in the proposal's state of
the art). This module only wraps the *official* split for the purpose of
getting a first working classifier; the leakage-aware evaluation policy
from WP2/WP4 should be layered on top before any result is used for model
selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment import (  # noqa: F401 - Cutout re-exported for callers
    AugmentationConfig,
    Cutout,
    DERMAMNIST_MEAN,
    DERMAMNIST_STD,
    build_eval_transform,
    build_train_transform,
)

try:
    import medmnist
    from medmnist import INFO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'medmnist' package is required. Install it with: pip install medmnist"
    ) from exc


DATA_FLAG = "dermamnist"


@dataclass
class DermaMNISTBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    task: str  # e.g. "multi-class"
    in_channels: int
    augmentation: Optional[AugmentationConfig] = None
    # Train-split label histogram, e.g. [228, 359, 769, 80, 779, 4693, 99].
    # Needed by any class-imbalance correction, so it travels with the data
    # rather than being hard-coded in the caller.
    class_counts: list[int] = field(default_factory=list)
    balanced_sampler: bool = False


def build_transforms(train: bool, aug: Optional[AugmentationConfig] = None):
    """Returns the transform pipeline for one split.

    `train=False` always returns the deterministic eval pipeline and ignores
    `aug`. `train=True` with `aug=None` also returns the eval pipeline, i.e.
    training without augmentation.
    """
    if not train:
        return build_eval_transform()
    if aug is None:
        return build_eval_transform()
    return build_train_transform(aug)


def _labels_of(dataset) -> np.ndarray:
    """Flat int array of a split's labels, without running the transforms.

    medmnist exposes `.labels` as an (N, 1) array; the fallback covers any
    dataset that does not, at the cost of decoding every image once.
    """
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = [dataset[i][1] for i in range(len(dataset))]
    return np.asarray(labels).reshape(-1).astype(np.int64)


def get_dermamnist_loaders(batch_size: int = 128, num_workers: int = 4,
                            aug: Optional[AugmentationConfig] = None,
                            data_root: str | None = None,
                            balanced_sampler: bool = False) -> DermaMNISTBundle:
    """Builds train/val/test DataLoaders for DermaMNIST.

    Args:
        aug: augmentation applied to the *train* split only. `None` means no
            augmentation. Build one with `fpvit.augment.preset(name)` or
            `fpvit.augment.resolve_augmentation(...)`.
        balanced_sampler: draw training samples with replacement, with
            probability inversely proportional to class frequency, so each
            epoch is class-balanced in expectation. An alternative to the
            loss weighting of `engine.compute_class_weights` - using both at
            once corrects the imbalance twice. Val and test are never
            resampled. The sampler draws from the global torch RNG, so it is
            covered by the seed and by `train.py --resume`.

    Downloads the dataset (first call only) via the medmnist package's own
    mirror. Requires outbound network access; if that is unavailable in your
    environment, pre-download the .npz file and pass `data_root` pointing to
    the folder that contains it.
    """
    info = INFO[DATA_FLAG]
    num_classes = len(info["label"])
    task = info["task"]
    in_channels = info["n_channels"]

    DataClass = getattr(medmnist, info["python_class"])

    # medmnist raises if `root` is None, so only override its default when a
    # folder was actually requested.
    root_kwargs = {"root": data_root} if data_root is not None else {}

    train_set = DataClass(split="train", transform=build_transforms(True, aug),
                           download=True, **root_kwargs)
    val_set = DataClass(split="val", transform=build_transforms(False),
                         download=True, **root_kwargs)
    test_set = DataClass(split="test", transform=build_transforms(False),
                          download=True, **root_kwargs)

    train_labels = _labels_of(train_set)
    class_counts = np.bincount(train_labels, minlength=num_classes).tolist()

    sampler = None
    if balanced_sampler:
        counts = np.asarray(class_counts, dtype=np.float64)
        per_class = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(per_class[train_labels], dtype=torch.double),
            num_samples=len(train_set),
            replacement=True,
        )

    # shuffle and sampler are mutually exclusive in DataLoader; the sampler
    # already draws in random order.
    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=(sampler is None), sampler=sampler,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return DermaMNISTBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        task=task,
        in_channels=in_channels,
        augmentation=aug,
        class_counts=class_counts,
        balanced_sampler=balanced_sampler,
    )
