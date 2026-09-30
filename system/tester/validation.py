"""
The ensemble's common validation set: DermaMNIST-C val.

Every weight the ensemble uses - the soft-vote skill, the hard-vote precision
per class, the classes a model is blind to - and the promotion gate's
with/without comparison are validation numbers. They are only comparable if
every member is scored on the same images, whatever split it was trained on
and whatever its input size. The checkpoint's own `val_metrics` do not
qualify: a 28 px model of the official split was scored on the official val
(which leaks lesions from train), a 224 px model on C-val.

So each member is scored here on DermaMNIST-C val, at its own input size and
normalisation (28 px from dermamnist_corrected_28.npz, 224 px from the 224
release; both hold the same 573 images, resized from the originals).

Leakage: C-val holds only lesions that are not in C-train, and C-train
contains the whole official train split, so C-val is clean for models of the
official split and of DermaMNIST-C. It is NOT clean for a model trained on
DermaMNIST-E, whose train split is all of HAM10000: such models are refused.
One bias remains and is stated: C-val is a subset of the official val split,
on which the official-split models chose their best epoch, so their C-val
numbers are slightly optimistic. The 224 px models get no such advantage.

The result is cached next to the checkpoint in `val_dermamnist_c.json`,
keyed by the checkpoint's SHA-256, together with the per-image
probabilities, so the promotion gate never has to run a model twice.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader

from nets.dataset import eval_transform_for, load_split, model_input
from nets.engine import resolve_device
from nets.zoo import build_from_config

import paths

VALIDATION_SET = "dermamnist_c"
SIDECAR = f"val_{VALIDATION_SET}.json"
_NUM_CLASSES = 7


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def trained_on(cfg: dict) -> str:
    return cfg.get("dataset") or "dermamnist"


@torch.no_grad()
def _score(ckpt: dict, device) -> tuple[np.ndarray, np.ndarray]:
    cfg = ckpt["config"]
    img_size, _ = model_input(cfg)
    ds = load_split("val", eval_transform_for(cfg), VALIDATION_SET, img_size, str(paths.DATASET))
    model = build_from_config(cfg, num_classes=_NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    probs = torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in loader]).numpy()
    return probs, ds.labels.reshape(-1).astype(np.int64)


def _metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    labels = list(range(_NUM_CLASSES))
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    try:
        auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        auc = None
    return {"balanced_acc": float(balanced_accuracy_score(y, pred)),
            "acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "macro_auc": auc,
            "per_class_precision": [float(v) for v in precision],
            "per_class_recall": [float(v) for v in recall]}


def validation_record(ckpt_path: str | Path, device: str = "cpu", ckpt: dict | None = None) -> dict:
    """The checkpoint's metrics and per-image probabilities on DermaMNIST-C val.

    Read from the sidecar when it matches the checkpoint's SHA-256, computed
    (and cached) otherwise. Downloads the C-val data on first use. Raises
    ValueError for a model trained on DermaMNIST-E.
    """
    path = Path(ckpt_path)
    sidecar = path.with_name(SIDECAR)
    sha = _sha256(path)
    if sidecar.exists():
        try:
            record = json.loads(sidecar.read_text())
            if record.get("checkpoint_sha256") == sha:
                return record
        except (OSError, json.JSONDecodeError):
            pass                                        # unreadable cache: recompute

    ckpt = ckpt or torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if trained_on(cfg) == "dermamnist_e":
        raise ValueError("trained on DermaMNIST-E, whose train split contains every image of "
                         "the common validation set (DermaMNIST-C val)")
    probs, y = _score(ckpt, resolve_device(device))
    img_size, norm = model_input(cfg)
    record = {
        "validation_set": VALIDATION_SET,
        "checkpoint_sha256": sha,
        "trained_on": trained_on(cfg),
        "img_size": img_size,
        "norm": norm,
        "n_images": int(len(y)),
        "metrics": _metrics(y, probs),
        "labels": y.tolist(),
        "probs": np.round(probs, 6).tolist(),
    }
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, sidecar)
    return record
