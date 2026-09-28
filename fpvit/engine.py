"""
Train / evaluate loops for FPViT on DermaMNIST.

Metrics:
  - ACC and macro one-vs-rest AUROC, as used in the FPViT paper (Section IV-B),
    matching the standard MedMNIST evaluation protocol.
  - macro-F1, balanced accuracy, per-class precision/recall/F1 and the
    confusion matrix, as required by the AgenticDerma project proposal's
    Evaluation Agent (Section 4.4) for model comparison beyond a single
    accuracy number.

Class weighting
---------------
`compute_class_weights` turns the train-split class counts into a per-class
weight for `CrossEntropyLoss`. It is applied to the **training loss only**:
`evaluate` always uses the unweighted loss, so `val_loss` stays comparable
across runs that use different weighting schemes (and remains usable as a
selection metric). Train loss, by contrast, is *not* comparable between
schemes - compare those runs on the validation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


@dataclass
class TrainMetrics:
    """What one training epoch reports back.

    Train accuracy is tracked alongside the loss so the train/val gap (i.e.
    overfitting, which the paper's Cutout is meant to control) is visible in
    the logs rather than having to be guessed from the loss alone.
    """

    loss: float
    acc: float

    def summary(self) -> str:
        return f"loss={self.loss:.4f} acc={self.acc:.4f}"


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    macro_auc: Optional[float]
    macro_f1: float
    balanced_acc: float
    per_class_precision: np.ndarray = field(repr=False, default=None)
    per_class_recall: np.ndarray = field(repr=False, default=None)
    per_class_f1: np.ndarray = field(repr=False, default=None)
    confusion: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        auc_str = f"{self.macro_auc:.4f}" if self.macro_auc is not None else "n/a"
        return (f"loss={self.loss:.4f} acc={self.acc:.4f} macroAUC={auc_str} "
                f"macroF1={self.macro_f1:.4f} balAcc={self.balanced_acc:.4f}")


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Picks the best available accelerator when `requested` is None/"auto".

    Adds Apple-Silicon (MPS) support, which the original cuda-or-cpu default
    silently skipped on macOS.
    """
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _flatten_labels(labels: torch.Tensor) -> torch.Tensor:
    # medmnist single-label tasks return labels shaped (batch, 1); flatten to (batch,)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    return labels.long()


def compute_class_weights(class_counts, scheme: str = "none",
                           beta: float = 0.999) -> Optional[torch.Tensor]:
    """Per-class loss weights from the train-split counts.

    DermaMNIST's train split is [228, 359, 769, 80, 779, 4693, 99]: class 5 is
    67 % of the data and class 3 has 80 images, so the unweighted loss has
    little reason to ever predict class 3. These are the two standard
    corrections:

      "inverse"    w_c = N / (K * n_c) - the textbook inverse-frequency
                   weight. Aggressive: here it hands class 3 a weight ~59x
                   that of class 5, which can destabilise early training.
      "effective"  w_c = (1 - beta) / (1 - beta^n_c), the effective-number
                   weighting of Cui et al., CVPR 2019. `beta` interpolates
                   between no weighting (beta=0) and inverse frequency
                   (beta->1); at the default 0.999 the class 3 / class 5
                   ratio is ~13x instead of ~59x.

    Both are rescaled to mean 1 across classes, so the overall loss magnitude
    (and therefore the usable learning-rate range) stays comparable to an
    unweighted run. Classes with zero samples get weight 0.

    Returns None for scheme="none", which is what `nn.CrossEntropyLoss`
    expects for "no weighting".
    """
    if scheme == "none":
        return None

    counts = np.asarray(class_counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts)

    if scheme == "inverse":
        n_total, n_present = counts.sum(), int(present.sum())
        weights[present] = n_total / (n_present * counts[present])
    elif scheme == "effective":
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    else:
        raise ValueError(f"unknown class-weight scheme: {scheme!r}")

    mean = weights[present].mean()
    if mean > 0:
        weights = weights / mean
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model: nn.Module, loader, optimizer, device, epoch: int,
                     log_every: int = 50, progress: bool = True,
                     class_weights: Optional[torch.Tensor] = None,
                     scaler: Optional["torch.amp.GradScaler"] = None,
                     grad_clip: float = 0.0,
                     label_smoothing: float = 0.0) -> TrainMetrics:
    """One pass over the train split.

    scaler: a CUDA GradScaler turns on mixed precision (fp16 autocast), which
        roughly halves memory and time at 224 px on a T4. None = full fp32.
    grad_clip: max global gradient norm (0 = off); measured on the unscaled
        gradients when mixed precision is on.
    label_smoothing: CrossEntropyLoss's label_smoothing, training side only.
    """
    model.train()
    # Weighted only on the training side - see the module docstring.
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    amp = scaler is not None
    running_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_seen = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False) if progress else None
    for images, labels in (pbar if pbar is not None else loader):
        labels = _flatten_labels(labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits.float(), labels)
        if amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        with torch.no_grad():
            n_correct += int((logits.argmax(dim=1) == labels).sum())
            n_seen += int(labels.numel())

        if pbar is not None and n_batches % log_every == 0:
            pbar.set_postfix(loss=running_loss / n_batches,
                             acc=n_correct / max(n_seen, 1))

    return TrainMetrics(
        loss=running_loss / max(n_batches, 1),
        acc=n_correct / max(n_seen, 1),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, num_classes: int,
              progress: bool = True) -> EpochMetrics:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_probs, all_preds, all_labels = [], [], []
    running_loss, n_batches = 0.0, 0

    batches = tqdm(loader, desc="eval", leave=False) if progress else loader
    for images, labels in batches:
        labels = _flatten_labels(labels)
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    conf = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    macro_auc = None
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                   labels=list(range(num_classes)))
    except ValueError:
        # Happens if a class is entirely absent from a small/synthetic eval batch.
        macro_auc = None

    return EpochMetrics(
        loss=running_loss / max(n_batches, 1),
        acc=acc,
        macro_auc=macro_auc,
        macro_f1=macro_f1,
        balanced_acc=balanced_acc,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        confusion=conf,
    )
