"""
Local FPViT checkpoints as a model zoo: discover, describe, load, predict.

Framework-neutral on purpose - no LangGraph / LangChain import here. The graph
in `graph.py` calls these functions from its nodes; a different orchestration
framework would call the same ones.

Every `best_model.pt` written by `train.py` already carries what an ensemble
needs to weigh a model's opinion: its validation metrics at the selected epoch,
including per-class precision and recall. `ModelCard` reads them once, so the
voting step can know that e.g. a model with recall 0.0 on dermatofibroma never
predicts it, and that a model's "melanoma" vote is right X% of the time on val.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from fpvit.dataset import build_transforms
from fpvit.engine import resolve_device
from fpvit.zoo import build_from_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where train.py (by hand, via simple_agent, via the campaign) leaves its runs.
DEFAULT_MODEL_ROOTS = [
    PROJECT_ROOT / "runs",
    PROJECT_ROOT / "runs_agent",
    PROJECT_ROOT / "runs test solo training pyramid",
    # Models the training agent produced AND a human promoted (train_agent/promotion.py).
    # Candidates still under evaluation live in runs_train/, which is not scanned.
    PROJECT_ROOT / "models_promoted",
]

# Same order as medmnist INFO["dermamnist"]["label"]; hard-coded so that
# building the zoo does not need medmnist's metadata at import time.
CLASS_NAMES = [
    "actinic keratoses / intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]
NUM_CLASSES = len(CLASS_NAMES)
CHANCE_BALANCED_ACC = 1.0 / NUM_CLASSES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def collect_images(targets: Iterable[str | Path]) -> list[Path]:
    """Expand files and folders into a sorted, de-duplicated list of images."""
    out: list[Path] = []
    for target in targets:
        p = Path(target).expanduser()
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES))
        elif p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
        else:
            raise FileNotFoundError(f"not an image or a folder of images: {p}")
    seen, unique = set(), []
    for p in out:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


@dataclass
class ModelCard:
    """What the ensemble knows about one checkpoint, read from the checkpoint itself."""

    name: str                       # run directory name, unique within the zoo
    path: str
    epoch: int | None
    selection_metric: str | None
    val_balanced_acc: float
    val_macro_auc: float | None
    val_per_class_precision: list[float]
    val_per_class_recall: list[float]
    config: dict = field(default_factory=dict)

    @property
    def skill(self) -> float:
        """Balanced accuracy above chance - the weight of this model's soft vote.

        A model at chance level (1/7) gets ~0 say, whatever its confidence.
        Floored at a small epsilon so a weak model still shows up in the tally.
        """
        return max(self.val_balanced_acc - CHANCE_BALANCED_ACC, 0.01)

    @property
    def blind_classes(self) -> list[int]:
        """Classes this model never recovered on val (recall 0.0)."""
        return [c for c, r in enumerate(self.val_per_class_recall) if r == 0.0]

    def public(self) -> dict:
        d = asdict(self)
        d.pop("config")
        d["skill"] = round(self.skill, 4)
        d["blind_classes"] = self.blind_classes
        d["arch"] = self.config.get("arch") or "fpvit"
        d["optimizer"] = self.config.get("optimizer")
        d["class_weight"] = self.config.get("class_weight")
        return d


def _card_from_checkpoint(path: Path, ckpt: dict, name: str) -> ModelCard:
    vm = ckpt.get("val_metrics") or {}
    return ModelCard(
        name=name,
        path=str(path),
        epoch=ckpt.get("epoch"),
        selection_metric=ckpt.get("selection_metric"),
        val_balanced_acc=float(vm.get("balanced_acc", CHANCE_BALANCED_ACC)),
        val_macro_auc=float(vm["macro_auc"]) if "macro_auc" in vm else None,
        val_per_class_precision=[float(x) for x in vm.get("per_class_precision", [0.0] * NUM_CLASSES)],
        val_per_class_recall=[float(x) for x in vm.get("per_class_recall", [0.0] * NUM_CLASSES)],
        config=dict(ckpt.get("config") or {}),
    )


class ModelZoo:
    """Every `best_model.pt` under the given roots, loaded lazily and cached.

    Loading a checkpoint costs ~70 MB of RAM and a second or two; a supervisor
    that calls the predictor many times in one process pays it once.
    """

    def __init__(self, roots: Iterable[str | Path] | None = None,
                 device: str = "auto",
                 min_balanced_acc: float = 0.0,
                 exclude: Iterable[str] = ()):
        self.roots = [Path(r) for r in (roots or DEFAULT_MODEL_ROOTS)]
        self.device = resolve_device(device)
        self.min_balanced_acc = min_balanced_acc
        self.exclude = set(exclude)
        self._transform = build_transforms(train=False)
        self._models: dict[str, torch.nn.Module] = {}
        self.cards: dict[str, ModelCard] = {}
        self.skipped: list[dict] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the roots (a training agent may have produced a new run)."""
        self.cards.clear()
        self.skipped.clear()
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/best_model.pt")):
                name = path.parent.name
                if name in self.cards:                 # same run name in two roots
                    name = f"{root.name}/{name}"
                if name in self.exclude or path.parent.name in self.exclude:
                    self.skipped.append({"name": name, "reason": "excluded"})
                    continue
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as exc:              # a half-written checkpoint
                    self.skipped.append({"name": name, "reason": f"unreadable: {exc}"})
                    continue
                card = _card_from_checkpoint(path, ckpt, name)
                if card.val_balanced_acc < self.min_balanced_acc:
                    self.skipped.append({"name": name,
                                         "reason": f"val balanced_acc {card.val_balanced_acc:.3f} "
                                                   f"< {self.min_balanced_acc}"})
                    continue
                self.cards[name] = card
        self._models = {k: v for k, v in self._models.items() if k in self.cards}

    def _load(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            card = self.cards[name]
            ckpt = torch.load(card.path, map_location=self.device, weights_only=False)
            # Checkpoints without an "arch" key predate it and are FPViT.
            model = build_from_config(ckpt["config"], num_classes=NUM_CLASSES).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._models[name] = model
        return self._models[name]

    @torch.no_grad()
    def predict(self, images: list[Path]) -> dict[str, list[list[float]]]:
        """{model_name: [probabilities per image, in input order]}.

        Same eval-time preprocessing as `predict.py` and validation
        (ToTensor + Normalize). The models are built for 28x28 inputs, so any
        other size is resized here first; a large photo downsampled to 28x28 is
        out of distribution for DermaMNIST-trained models, and the report says so.
        """
        batch = []
        for path in images:
            img = Image.open(path).convert("RGB")
            if img.size != (28, 28):
                img = img.resize((28, 28), Image.BICUBIC)
            batch.append(self._transform(img))
        x = torch.stack(batch).to(self.device)

        out = {}
        for name in self.cards:
            probs = torch.softmax(self._load(name)(x), dim=1).cpu()
            out[name] = [[float(p) for p in row] for row in probs]
        return out
