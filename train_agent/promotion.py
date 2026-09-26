"""
Promotion gate: does a candidate make the prediction agent's ensemble better?

    evaluate_candidate()  soft-vote balanced accuracy on the VALIDATION split,
                          with the current ensemble and with the candidate
                          added, using the same skill-weighted soft vote as
                          predict_agent.voting. Only the val_* arrays of the
                          .npz are read: the test split is never loaded.
    promote()             copy the candidate into models_promoted/<new name>/
                          (a fresh folder, never overwritten; the original run
                          stays where it is) and append to the registry. The
                          prediction agent scans models_promoted/, so the model
                          votes from the next prediction on.

Caveat, stated in every evaluation: the weights of the vote are validation
metrics and the evaluation is on the same validation split, so the numbers are
optimistic in absolute terms. The *comparison* with/without is fair, because
both sides pay the same bias.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fpvit.dataset import build_transforms
from fpvit.engine import resolve_device
from fpvit.zoo import build_from_config
from predict_agent.models import CHANCE_BALANCED_ACC, CLASS_NAMES, PROJECT_ROOT, ModelZoo

PROMOTED = PROJECT_ROOT / "models_promoted"
REGISTRY = PROMOTED / "registry.jsonl"
MIN_GAIN = 0.005          # below this the "improvement" is validation noise
_VAL_CACHE: dict = {}


def _data_path() -> Path:
    return Path.home() / ".medmnist" / "dermamnist.npz"


def load_val() -> tuple[torch.Tensor, np.ndarray]:
    if "x" not in _VAL_CACHE:
        with np.load(_data_path()) as npz:              # lazy: only val_* are read
            images, labels = npz["val_images"], npz["val_labels"].reshape(-1)
        tf = build_transforms(train=False)
        _VAL_CACHE["x"] = torch.stack([tf(Image.fromarray(im)) for im in images])
        _VAL_CACHE["y"] = labels.astype(np.int64)
    return _VAL_CACHE["x"], _VAL_CACHE["y"]


@torch.no_grad()
def val_probs(ckpt_path: str, device) -> np.ndarray:
    key = ("probs", ckpt_path)
    if key not in _VAL_CACHE:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_from_config(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        x, _ = load_val()
        out = [torch.softmax(model(x[i:i + 256].to(device)), 1).cpu() for i in range(0, len(x), 256)]
        _VAL_CACHE[key] = torch.cat(out).numpy()
    return _VAL_CACHE[key]


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    recall = [float((pred[y == c] == c).mean()) if (y == c).any() else 0.0 for c in range(len(CLASS_NAMES))]
    return {"balanced_acc": round(float(np.mean(recall)), 4), "acc": round(float((pred == y).mean()), 4),
            "per_class_recall": dict(zip(CLASS_NAMES, [round(r, 3) for r in recall]))}


def _soft_vote(members: list[tuple[str, float]], device) -> np.ndarray:
    total = sum(w for _, w in members)
    return sum(w * val_probs(p, device) for p, w in members) / total


def evaluate_candidate(candidate: dict, device: str = "auto") -> dict:
    dev = resolve_device(device)
    zoo = ModelZoo(device="cpu")
    ckpt = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=False)
    cand_bal = float(ckpt.get("val_metrics", {}).get("balanced_acc", CHANCE_BALANCED_ACC))
    cand_skill = max(cand_bal - CHANCE_BALANCED_ACC, 0.01)

    current = [(c.path, c.skill) for c in zoo.cards.values()]
    _, y = load_val()
    cand_pred = val_probs(candidate["checkpoint"], dev).argmax(1)
    result = {
        "candidate": candidate["run"], "candidate_alone": _metrics(cand_pred, y),
        "candidate_skill": round(cand_skill, 4), "ensemble_size": len(current),
        "caveat": "vote weights and evaluation both come from the validation split: absolute "
                  "numbers are optimistic, the with/without comparison is fair",
    }
    if current:
        before = _metrics(_soft_vote(current, dev).argmax(1), y)
        after = _metrics(_soft_vote(current + [(candidate["checkpoint"], cand_skill)], dev).argmax(1), y)
    else:
        before, after = {"balanced_acc": 0.0, "acc": 0.0, "per_class_recall": {}}, result["candidate_alone"]
    delta = round(after["balanced_acc"] - before["balanced_acc"], 4)
    result.update(ensemble_now=before, ensemble_with_candidate=after, delta_balanced_acc=delta,
                  improves=delta >= MIN_GAIN,
                  recall_changes={c: round(after["per_class_recall"].get(c, 0) - before["per_class_recall"].get(c, 0), 3)
                                  for c in CLASS_NAMES})
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def promote(candidate: dict, evaluation: dict, campaign_id: str, decision: dict) -> dict:
    src = Path(candidate["dir"])
    name = re.sub(r"[^\w.-]+", "_", f"{candidate['arch']}_{campaign_id}_{candidate['run']}")
    dst = PROMOTED / name
    if dst.exists():
        raise FileExistsError(f"{dst} already exists: promoted models are never overwritten")
    dst.mkdir(parents=True)
    for f in ("best_model.pt", "experiment_record.json", "metrics.csv"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    entry = {
        "name": name, "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_run": str(src.relative_to(PROJECT_ROOT)), "arch": candidate["arch"],
        "sha256": _sha256(dst / "best_model.pt"), "val_at_best": candidate.get("val_at_best"),
        "ensemble_evaluation": {k: evaluation.get(k) for k in ("ensemble_now", "ensemble_with_candidate",
                                                              "delta_balanced_acc", "improves")},
        "human_decision": decision,
    }
    with open(REGISTRY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
