"""
DermaMNIST data loading for the FPViT baseline.

Uses the official `medmnist` package (MedMNIST v2), which ships the fixed
train/val/test split used for all MedMNIST Classification Decathlon
benchmarks (train 7007 / val 1003 / test 2005).

Augmentation lives in `nets/augment.py` and is described by an
`AugmentationConfig`, which this module simply applies to the train split.
Val and test are never augmented. Pass `aug=None` to train without any
augmentation.

Leakage-free alternatives (`dataset=` below)
--------------------------------------------
The official split was made per *image*, but HAM10000 holds several images
of the same lesion, so the same lesion sits in train, val and test
(Abhishek, Jain & Hamarneh, Scientific Data 2025). The authors release two
corrected datasets, as MedMNIST-format .npz files at 28 and 224 px, the 224
ones resized directly from the 600x450 originals (bicubic) rather than
upsampled from 28:

    dermamnist_c   DermaMNIST-C: every image of a lesion that is in train is
                   moved from val/test into train (8215 / 573 / 1227)
    dermamnist_e   DermaMNIST-E: train = all of HAM10000, val/test = the ISIC
                   2018 validation and test sets (10015 / 193 / 1511)

A model trained on C's train split can be tested on both C's and E's test
splits: E's are other lesions entirely, which makes them an external test.
The reverse does not hold: E's train split is all of HAM10000, so it
contains every image of C's (and the official) val and test splits.
Both are downloaded from Zenodo on first use and checked against their MD5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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

# Every dataset below uses the same 7 classes in the same order as DermaMNIST.
DATASETS = ("dermamnist", "dermamnist_c", "dermamnist_e")
DEFAULT_DATASET = "dermamnist"

# Zenodo record of Abhishek et al. (v1.1): file name -> MD5.
_ZENODO_RECORD = "12739457"
_NPZ_FILES = {
    ("dermamnist_c", 28): ("dermamnist_corrected_28.npz", "96e2862980877d45d9cacb5c25de6950"),
    ("dermamnist_c", 224): ("dermamnist_corrected_224.npz", "84920fb70c83b234c295b6f0d4ae2bc0"),
    ("dermamnist_e", 28): ("dermamnist_extended_28.npz", "ecacbf37d9cc79340226bd19bd4c7463"),
    ("dermamnist_e", 224): ("dermamnist_extended_224.npz", "b59d6a78a036a0bb48a1eff13d94333c"),
}


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


def build_transforms(train: bool, aug: Optional[AugmentationConfig] = None,
                     img_size: int = 28, norm: str = "dermamnist"):
    """Returns the transform pipeline for one split.

    `train=False` always returns the deterministic eval pipeline and ignores
    `aug`. `train=True` with `aug=None` also returns the eval pipeline, i.e.
    training without augmentation.
    """
    if not train or aug is None:
        return build_eval_transform(img_size, norm)
    return build_train_transform(aug, img_size, norm)


def model_input(cfg: dict) -> tuple[int, str]:
    """(input size, normalisation) a checkpoint's config was trained with.

    Configs from before `--img-size` / `--norm` existed are 28 px, DermaMNIST
    statistics.
    """
    return int(cfg.get("img_size") or 28), cfg.get("norm") or "dermamnist"


def eval_transform_for(cfg: dict):
    """The inference pipeline for a checkpoint: resize to its input size, then
    its normalisation. Every caller that feeds a stored model goes through
    here, so a 224 px model is never handed a 28 px image or vice versa."""
    return build_eval_transform(*model_input(cfg))


def dataset_file(dataset: str, img_size: int, data_root: str | None = None,
                 download: bool = True) -> Path:
    """Local path of a DermaMNIST-C/E .npz, downloading it (MD5-checked) if missing."""
    if (dataset, img_size) not in _NPZ_FILES:
        sizes = sorted(s for d, s in _NPZ_FILES if d == dataset)
        raise ValueError(f"{dataset} is released at {sizes} px, not {img_size}")
    name, md5 = _NPZ_FILES[(dataset, img_size)]
    root = Path(data_root).expanduser() if data_root else Path.home() / ".medmnist"
    path = root / name
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found")
        root.mkdir(parents=True, exist_ok=True)
        _download(f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{name}?download=1", path, md5)
    return path


def _ssl_context():
    """Verified TLS. The python.org macOS build ships without CA certificates
    ("Install Certificates.command"); certifi's bundle fills the gap when present."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(url: str, path: Path, md5: str) -> None:
    """Streams `url` to `path`, checking the MD5 before the file appears.

    Written to `<path>.part` and renamed only once the checksum matches, so an
    interrupted or corrupted download never leaves a file that looks complete.
    """
    import hashlib
    import os
    import urllib.request

    from tqdm import tqdm

    tmp = path.with_name(path.name + ".part")
    digest = hashlib.md5()
    context = _ssl_context() if url.startswith("https:") else None
    with urllib.request.urlopen(url, timeout=60, context=context) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            for block in iter(lambda: response.read(1 << 20), b""):
                f.write(block)
                digest.update(block)
                bar.update(len(block))
    if digest.hexdigest() != md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name}: MD5 {digest.hexdigest()} != expected {md5}; download discarded")
    os.replace(tmp, path)


class NpzSplit(Dataset):
    """One split of a MedMNIST-format .npz, returning what medmnist returns:
    (transformed PIL image, label array of shape (1,)).

    Only this split's two arrays are read from the archive; np.load
    decompresses them into memory once (the 224 px train split of
    DermaMNIST-C is ~1.2 GB).
    """

    def __init__(self, path: Path, split: str, transform=None):
        with np.load(path) as npz:
            self.imgs = npz[f"{split}_images"]
            self.labels = npz[f"{split}_labels"].reshape(-1, 1).astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]


def load_split(split: str, transform, dataset: str = DEFAULT_DATASET, img_size: int = 28,
               data_root: str | None = None) -> Dataset:
    """One split of any dataset in DATASETS, at the given size."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    if dataset == "dermamnist":
        info = INFO[DATA_FLAG]
        DataClass = getattr(medmnist, info["python_class"])
        # medmnist raises if `root` is None, so only override its default when a
        # folder was actually requested. size=28 is the original release; the
        # larger MedMNIST+ sizes keep the same (per-image, leaky) split.
        kwargs = {"root": data_root} if data_root is not None else {}
        if img_size != 28:
            kwargs["size"] = img_size
        return DataClass(split=split, transform=transform, download=True, **kwargs)
    return NpzSplit(dataset_file(dataset, img_size, data_root), split, transform)


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
                            balanced_sampler: bool = False,
                            dataset: str = DEFAULT_DATASET,
                            img_size: int = 28,
                            norm: str = "dermamnist") -> DermaMNISTBundle:
    """Builds train/val/test DataLoaders for DermaMNIST or DermaMNIST-C/E.

    Args:
        dataset: one of DATASETS; see the module docstring.
        img_size: 28, or 224 (64/128 too for the official split via medmnist).
        norm: input normalisation, a key of augment.NORMALIZATIONS.
        aug: augmentation applied to the *train* split only. `None` means no
            augmentation. Build one with `nets.augment.preset(name)` or
            `nets.augment.resolve_augmentation(...)`.
        balanced_sampler: draw training samples with replacement, with
            probability inversely proportional to class frequency, so each
            epoch is class-balanced in expectation. An alternative to the
            loss weighting of `engine.compute_class_weights` - using both at
            once corrects the imbalance twice. Val and test are never
            resampled. The sampler draws from the global torch RNG, so it is
            covered by the seed and by `train.py --resume`.

    Downloads the dataset (first call only): the official split via the
    medmnist package's own mirror, DermaMNIST-C/E from Zenodo. Requires
    outbound network access; if that is unavailable in your environment,
    pre-download the .npz file and pass `data_root` pointing to the folder
    that contains it.
    """
    info = INFO[DATA_FLAG]
    num_classes = len(info["label"])
    task = info["task"]
    in_channels = info["n_channels"]

    if data_root is not None:
        Path(data_root).mkdir(parents=True, exist_ok=True)

    def split(name: str, train: bool = False):
        return load_split(name, build_transforms(train, aug, img_size, norm),
                          dataset, img_size, data_root)

    train_set = split("train", train=True)
    val_set = split("val")
    test_set = split("test")

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
    # Workers stay alive between epochs: at 224 px, re-forking them (and
    # re-sharing a 1.2 GB array) every epoch is a visible cost.
    loader_kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
                     "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=(sampler is None), sampler=sampler,
                               drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

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
