"""Caricamento di DermaMNIST e preparazione dei batch.

Il dataset intero pesa 23,6 MB in uint8 e sta in RAM: viene caricato una volta sola e messo
in cache a livello di modulo. Le immagini restano uint8 fino al momento del batch, dove
conversione, resize e normalizzazione avvengono in blocco sul device (molto piu' veloce che
per singola immagine sulla CPU).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from d4_augmentation import random_d4

from .config import (
    CLASS_ABBR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NPZ_PATH,
    NUM_CLASSES,
    TrainConfig,
    load_eda_summary,
)

_SPLITS_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def load_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Restituisce {split: (images uint8 (N,28,28,3), labels int64 (N,))}, con cache."""
    global _SPLITS_CACHE
    if _SPLITS_CACHE is None:
        if not NPZ_PATH.exists():
            raise FileNotFoundError(
                f"Dataset non trovato in {NPZ_PATH}. Scaricalo con:\n"
                "  python -c \"import medmnist; medmnist.DermaMNIST(split='train', download=True)\""
            )
        npz = np.load(NPZ_PATH)
        _SPLITS_CACHE = {
            s: (npz[f"{s}_images"], npz[f"{s}_labels"].ravel().astype(np.int64))
            for s in ("train", "val", "test")
        }
    return _SPLITS_CACHE


def sample_image(class_name: str, index: int = 0, split: str = "train") -> np.ndarray:
    """Restituisce l'immagine `index`-esima di una classe, come array uint8 (28,28,3).

    Serve a ispezionare il dataset senza doverne prima esportare a mano dei campioni.

    Args:
        class_name: sigla della classe, una fra quelle in CLASS_ABBR.
        index: posizione all'interno della classe, contando dall'inizio dello split.
        split: 'train', 'val' oppure 'test'.

    Raises:
        ValueError: con un messaggio che elenca i valori ammessi, cosi' che chi chiama
            (agente incluso) possa correggersi senza consultare il codice.
    """
    name = class_name.strip().lower()
    if name not in CLASS_ABBR:
        raise ValueError(
            f"Classe '{class_name}' sconosciuta. Ammesse: {', '.join(CLASS_ABBR)}."
        )
    if split not in ("train", "val", "test"):
        raise ValueError(f"Split '{split}' sconosciuto. Ammessi: train, val, test.")

    images, labels = load_splits()[split]
    positions = np.flatnonzero(labels == CLASS_ABBR.index(name))
    if index < 0 or index >= len(positions):
        raise ValueError(
            f"Indice {index} fuori intervallo: la classe '{name}' ha {len(positions)} "
            f"immagini nello split '{split}' (indici da 0 a {len(positions) - 1})."
        )
    return images[positions[index]]


class DermaDataset(Dataset):
    """Immagini uint8 con augmentation opzionale applicata per campione.

    L'augmentation D4 riusa `random_d4` di `d4_augmentation.py`: e' una permutazione esatta
    dei pixel, quindi puo' essere applicata sull'array uint8 senza perdita.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, augmentation: str = "none", seed: int = 0):
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.augmentation == "d4":
            img, _ = random_d4(img, rng=self.rng)
        return torch.from_numpy(np.ascontiguousarray(img)), int(self.labels[idx])


def resolve_normalization(cfg: TrainConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Sceglie le costanti di normalizzazione secondo la config.

    Con pesi pre-addestrati vanno usate quelle di ImageNet: sono le statistiche su cui la
    rete e' stata addestrata, e cambiarle sposta gli input fuori dal dominio atteso dai pesi.
    Le costanti specifiche di DermaMNIST (stimate sul solo train dall'EDA) hanno senso quando
    si addestra da zero.
    """
    if cfg.normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    norm = load_eda_summary()["normalization"]
    return tuple(norm["mean"]), tuple(norm["std"])


def make_loaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    """Costruisce i DataLoader per i tre split ufficiali.

    L'augmentation e' applicata al solo training set: val e test devono restare intatti,
    altrimenti le metriche non sono confrontabili fra run.
    """
    splits = load_splits()
    loaders = {}
    for name, (images, labels) in splits.items():
        is_train = name == "train"
        ds = DermaDataset(
            images,
            labels,
            augmentation=cfg.augmentation if is_train else "none",
            seed=cfg.seed,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=0,          # il dataset e' gia' in RAM: i worker aggiungerebbero solo overhead
            drop_last=False,
        )
    return loaders


def prepare_batch(
    images: torch.Tensor,
    input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """uint8 (B,28,28,3) -> float normalizzato (B,3,S,S) sul device.

    Il resize avviene sul device e in batch. A 224 px l'upsampling da 28 px produce immagini
    intrinsecamente sfocate: non aggiunge dettaglio, serve a portare l'input nella scala per
    cui i pesi ImageNet sono stati addestrati.
    """
    # `.contiguous()` non e' cosmetico: `permute` restituisce una vista con stride non
    # contigui che sopravvive alla forward ma fa fallire il backward di ResNet
    # ("view size is not compatible with input tensor's size and stride") sul flatten finale.
    x = images.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    if input_size != x.shape[-1]:
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return (x - mean) / std


def class_weights_tensor(cfg: TrainConfig, device: torch.device) -> torch.Tensor | None:
    """Pesi per classe da passare alla loss, letti dal riepilogo EDA.

    Restituisce None se class_weighting == "none".
    """
    if cfg.class_weighting == "none":
        return None
    weights = load_eda_summary()["class_weights_balanced"]
    if len(weights) != NUM_CLASSES:
        raise ValueError(
            f"Il riepilogo EDA contiene {len(weights)} class weight, attesi {NUM_CLASSES}."
        )
    return torch.tensor(weights, dtype=torch.float32, device=device)
