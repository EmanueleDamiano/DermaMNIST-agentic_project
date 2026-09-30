"""
Local FPViT checkpoints as a model zoo: discover, describe, load, predict.

Framework-neutral on purpose - no LangGraph / LangChain import here. The graph
in `graph.py` calls these functions from its nodes; a different orchestration
framework would call the same ones.

What an ensemble needs to weigh a model's opinion is its validation metrics,
including per-class precision and recall, so the voting step can know that e.g.
a model with recall 0.0 on dermatofibroma never predicts it, and that a model's
"melanoma" vote is right X% of the time on val. `ModelCard` reads them from the
COMMON validation set, DermaMNIST-C val (see `validation.py`), not from the
checkpoint's own `val_metrics`: models trained on different splits or at
different input sizes are only comparable on the same images.

Runs listed in `model/ensemble_exclusions.json` are left out, with
the reason recorded there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from nets.dataset import eval_transform_for, model_input
from nets.engine import resolve_device
from nets.zoo import build_from_config
from tester.validation import VALIDATION_SET, validation_record

import paths

PROJECT_ROOT = paths.ROOT

# Folders whose <run>/best_model.pt form the prediction ensemble.
BASELINE_DIR = paths.BASELINE      # shipped with the repository
PROMOTED_DIR = paths.PROMOTED      # trained by the agent AND approved by a human
DEFAULT_MODEL_ROOTS = [
    BASELINE_DIR,
    PROMOTED_DIR,
    # Manual train.py runs. Candidates of a training campaign live in output/campaigns/,
    # which is not scanned: they vote only after a human promotes them.
    paths.RUNS,
]

# Run name -> why it is not in the ensemble. Human-edited; the testing agent,
# the promotion gate and the platform all read it.
EXCLUSIONS_FILE = paths.EXCLUSIONS


def load_exclusions(path: Path = EXCLUSIONS_FILE) -> dict[str, str]:
    try:
        return {str(k): str(v) for k, v in json.loads(path.read_text()).items()}
    except FileNotFoundError:
        return {}

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
    """What the ensemble knows about one checkpoint.

    The val_* fields are measured on the common validation set (`val_set`),
    not taken from the checkpoint, so every card is comparable.
    """

    name: str                       # run directory name, unique within the zoo
    path: str
    epoch: int | None
    selection_metric: str | None
    val_balanced_acc: float
    val_macro_auc: float | None
    val_per_class_precision: list[float]
    val_per_class_recall: list[float]
    val_set: str = VALIDATION_SET
    input_size: int = 28            # the model's own input resolution (px)
    trained_on: str = "dermamnist"
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


def _card_from_checkpoint(path: Path, ckpt: dict, name: str, val: dict) -> ModelCard:
    vm = val["metrics"]
    return ModelCard(
        name=name,
        path=str(path),
        epoch=ckpt.get("epoch"),
        selection_metric=ckpt.get("selection_metric"),
        val_balanced_acc=float(vm["balanced_acc"]),
        val_macro_auc=vm.get("macro_auc"),
        val_per_class_precision=[float(x) for x in vm["per_class_precision"]],
        val_per_class_recall=[float(x) for x in vm["per_class_recall"]],
        val_set=val["validation_set"],
        input_size=int(val["img_size"]),
        trained_on=val["trained_on"],
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
                 exclude: Iterable[str] = (),
                 exclusions: dict[str, str] | None = None):
        """exclude: extra run names to leave out; exclusions: name -> reason,
        default read from ensemble_exclusions.json."""
        self.roots = [Path(r) for r in (roots or DEFAULT_MODEL_ROOTS)]
        self.device = resolve_device(device)
        self.min_balanced_acc = min_balanced_acc
        self.exclusions = load_exclusions() if exclusions is None else dict(exclusions)
        self.exclusions.update({n: "excluded by the caller" for n in exclude})
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
                reason = self.exclusions.get(name) or self.exclusions.get(path.parent.name)
                if reason:
                    self.skipped.append({"name": name, "reason": f"excluded: {reason}"})
                    continue
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as exc:              # a half-written checkpoint
                    self.skipped.append({"name": name, "reason": f"unreadable: {exc}"})
                    continue
                try:
                    val = validation_record(path, self.device, ckpt)
                except Exception as exc:              # E-trained, or C-val unavailable
                    self.skipped.append({"name": name, "reason": f"no common validation: {exc}"})
                    continue
                card = _card_from_checkpoint(path, ckpt, name, val)
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

        Same eval-time preprocessing as `predict.py` and validation, per model:
        each image is resized (bicubic) to the input size that model was
        trained at - 28, or 224 for a DermaMNIST-C/E model - and normalised
        with its statistics. One batch per distinct (size, normalisation), so
        the usual all-28 ensemble still preprocesses once. A large photo
        downsampled to 28x28 is out of distribution for DermaMNIST-trained
        models, and the report says so.
        """
        pil = [Image.open(path).convert("RGB") for path in images]
        batches: dict[tuple, torch.Tensor] = {}
        out = {}
        for name, card in self.cards.items():
            key = model_input(card.config)
            if key not in batches:
                tf = eval_transform_for(card.config)
                batches[key] = torch.stack([tf(img) for img in pil]).to(self.device)
            probs = torch.softmax(self._load(name)(batches[key]), dim=1).cpu()
            out[name] = [[float(p) for p in row] for row in probs]
        return out
