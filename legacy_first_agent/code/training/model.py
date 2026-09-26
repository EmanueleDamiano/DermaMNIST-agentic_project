"""Costruzione del classificatore: ResNet-18 pre-addestrata adattata a 7 classi."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .config import NUM_CLASSES


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    input_size: int = 224,
) -> nn.Module:
    """Restituisce una ResNet-18 con la testa sostituita per `num_classes`.

    Args:
        pretrained: carica i pesi ImageNet. Richiede il checkpoint in
            ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth.
        freeze_backbone: se True congela tutto tranne la testa (linear probe). Molto piu'
            veloce, ma le feature restano quelle di ImageNet: e' un baseline, non un
            fine tuning.
        input_size: dimensione dell'input. Se <= 64 lo stem viene adattato (vedi sotto).

    Nota sullo stem a bassa risoluzione: la ResNet standard apre con una convoluzione 7x7
    stride 2 seguita da un maxpool stride 2, che riducono l'input di un fattore 4 prima
    ancora del primo blocco residuo. Su un input di 28-64 px questo distrugge quasi tutta
    l'informazione spaziale, quindi per input_size <= 64 lo stem viene sostituito con una
    conv 3x3 stride 1 senza maxpool.

    Il prezzo di quella sostituzione va detto chiaramente: **i pesi pre-addestrati del primo
    layer vengono buttati** e reinizializzati a caso. E' il compromesso classico fra
    "sfruttare il pre-addestramento" (che vuole 224 px) e "non distruggere immagini piccole".
    Il default di progetto e' 224 px con stem intatto.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # La testa e' sempre nuova e sempre addestrabile: va sostituita DOPO l'eventuale freeze,
    # altrimenti verrebbe congelata anche lei e il modello non imparerebbe nulla.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Restituisce (parametri addestrabili, parametri totali)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def select_device() -> torch.device:
    """MPS su Apple Silicon, altrimenti CUDA, altrimenti CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
