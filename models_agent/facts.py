"""
The facts the models agent answers from, computed in code.

Framework-neutral on purpose (no LangGraph here), like predict_agent/models.py.
Everything is read from what the rest of the system already wrote:

    checkpoints            architecture, input size, training configuration
    experiment_record.json epochs run, stop reason
    val_dermamnist_c.json  per-image probabilities on the common validation set
                           (predict_agent/validation.py), from which every
                           metric, ROC curve and confusion matrix below is derived
    registry.jsonl         promotions and the human decisions behind them
    ensemble_exclusions.json  which runs are out of the ensemble, and why
    the .npz datasets      class counts of the train and validation splits

The TEST split is never read, not even its labels: the agents do not look at
the test set. The facts are validation facts, and say so.

Two outputs:
    facts   compact, JSON, what the LLM is given (and what its answer is checked against)
    charts  ROC points and confusion matrices, for the platform to draw; never sent to the LLM
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from fpvit.dataset import dataset_file
from fpvit.zoo import ARCHITECTURES, build_from_config
from predict_agent.models import (CHANCE_BALANCED_ACC, CLASS_NAMES, DEFAULT_MODEL_ROOTS,
                                  EXCLUSIONS_FILE, PROJECT_ROOT, ModelZoo, load_exclusions)
from predict_agent.validation import VALIDATION_SET, validation_record

CLASS_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
REGISTRY = PROJECT_ROOT / "models_promoted" / "registry.jsonl"
ROC_POINTS = 60                     # per curve, enough to draw it

# Static facts about the data and the rules. They restate what the code and the
# project documents fix; the numbers that can be computed are computed instead.
DATASET_NOTES = {
    "source": "All three datasets come from HAM10000 (dermoscopic images of 7 lesion classes); "
              "DermaMNIST-E also uses the ISIC 2018 challenge images for its validation and test splits.",
    "dermamnist_official": "Official MedMNIST split, made per image: several images of the same lesion "
                           "can sit in train, validation and test (Abhishek et al., Scientific Data 2025). "
                           "Sizes 7,007 / 1,003 / 2,005 images.",
    "dermamnist_c": "DermaMNIST-C (corrected): every image of a lesion that appears in the official train "
                    "split was moved from validation/test into train, so no lesion is shared across splits. "
                    "Sizes 8,215 / 573 / 1,227 images. Released at 28 and 224 px, the 224 px images resized "
                    "directly from the 600x450 originals (bicubic), not upscaled from 28 px.",
    "dermamnist_c_val": "The common validation set of the ensemble: 573 images, all taken from the official "
                        "validation split (the other 430 official validation images belong to lesions that "
                        "are in train, so they were moved to train). It holds no lesion of C's train split, "
                        "hence none of the official train split either.",
    "dermamnist_e": "DermaMNIST-E (extended): train = all 10,015 HAM10000 images, validation and test = the "
                    "ISIC 2018 validation (193) and test (1,511) sets, which share no image with HAM10000. "
                    "A model trained on E is never scored on C, because E's train contains all of C.",
    "why_c_val": "Every model is scored on the same images, at its own input size, so that vote weights and "
                 "promotion comparisons are comparable across models trained on different splits and "
                 "resolutions.",
}
RULES = {
    "vote_weight": "skill = balanced accuracy on C-val minus 1/7 (chance), floored at 0.01",
    "soft_vote": "probabilities averaged with the skill weights: the default decision",
    "hard_vote": "each model's top class, weighted by its C-val precision for that class",
    "resolution": "a model does not vote on an image smaller than its own input; 224 px models shown "
                  "upsampled 28 px images predict nevus for 95 % of them",
    "promotion": "a model enters the ensemble only if it raises the ensemble's C-val balanced accuracy by "
                 "at least 0.005 and a human approves",
    "exclusion": "a run is removed from the ensemble only by a human, with the reason in "
                 "ensemble_exclusions.json",
}
CAVEATS = [
    "C-val is small: 573 images, with only 5 dermatofibroma and 7 vascular lesion images, so per-class "
    "metrics for those classes move a lot with a single image.",
    "The ensemble's own metrics are optimistic: its vote weights come from the same C-val images.",
    "The official-split models chose their best epoch on the official validation split, of which C-val is a "
    "subset, so their C-val numbers are slightly optimistic; the 224 px models have no such advantage.",
    "These are validation results. The test split is never read by the agents.",
]


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _counts(dataset: str, split: str, img_size: int = 28) -> list[int] | None:
    """Class counts of a train/val split, reading only that split's labels."""
    if split == "test":
        raise ValueError("the agents never read the test split")
    try:
        if dataset == "dermamnist":
            path = Path.home() / ".medmnist" / "dermamnist.npz"
        else:
            path = dataset_file(dataset, img_size, download=False)
        with np.load(path) as z:
            return np.bincount(z[f"{split}_labels"].reshape(-1), minlength=7).tolist()
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    try:
        auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = None
    per_class = {}
    for c in labels:
        onehot = (y == c).astype(int)
        auc_c = roc_auc_score(onehot, p[:, c]) if 0 < onehot.sum() < len(y) else None
        per_class[CLASS_SHORT[c]] = {"precision": _r(prec[c]), "recall": _r(rec[c]), "f1": _r(f1[c]),
                                     "auc": _r(auc_c), "support": int(sup[c])}
    return {"accuracy": _r(accuracy_score(y, pred)), "balanced_accuracy": _r(balanced_accuracy_score(y, pred)),
            "macro_f1": _r(f1_score(y, pred, average="macro", zero_division=0)), "macro_auc": _r(auc),
            "predicted_as_nevus": _r((pred == 5).mean()), "per_class": per_class}


def _roc(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for c in range(len(CLASS_NAMES)):
        onehot = (y == c).astype(int)
        if not 0 < onehot.sum() < len(y):
            continue
        fpr, tpr, _ = roc_curve(onehot, p[:, c])
        if len(fpr) > ROC_POINTS:             # thin the curve, keeping its ends
            idx = np.unique(np.linspace(0, len(fpr) - 1, ROC_POINTS).round().astype(int))
            fpr, tpr = fpr[idx], tpr[idx]
        out[CLASS_SHORT[c]] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]
    return out


def _registry() -> dict[str, dict]:
    try:
        lines = REGISTRY.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    return {e["name"]: e for e in (json.loads(ln) for ln in lines if ln.strip())}


def _record(ckpt_path: str) -> dict:
    try:
        return json.loads((Path(ckpt_path).parent / "experiment_record.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _model_facts(card, role: str, reason: str, registry: dict, n_params: int | None) -> dict:
    cfg, rec = card.config, _record(card.path)
    arch = cfg.get("arch") or "fpvit"
    promo = registry.get(card.name)
    facts = {
        "name": card.name, "role": role,
        "architecture": arch, "architecture_description": ARCHITECTURES.get(arch, ""),
        "input_size_px": card.input_size, "trained_on": card.trained_on,
        "pretrained_imagenet": bool(cfg.get("pretrained")), "parameters": n_params,
        "training": {k: cfg.get(k) for k in ("optimizer", "lr", "weight_decay", "batch_size", "class_weight",
                                              "aug_preset", "select_on", "seed", "epochs")},
        "epochs_run": rec.get("epochs_completed"), "stop_reason": rec.get("stop_reason"),
        "selected_epoch": card.epoch,
        "vote_weight": _r(card.skill) if role == "ensemble member" else None,
        "blind_classes_on_c_val": [CLASS_SHORT[c] for c in card.blind_classes],
        "folder": str(Path(card.path).parent.relative_to(PROJECT_ROOT)),
    }
    if arch == "fpvit":
        facts["fpvit_shape"] = {k: cfg.get(k) for k in ("stem", "token_grid", "embed_dim", "depth", "num_heads")}
    if reason:
        facts["excluded_because"] = reason
    if promo:
        ev = promo.get("ensemble_evaluation") or {}
        facts["promotion"] = {"promoted_at": promo.get("promoted_at"), "source_run": promo.get("source_run"),
                              "ensemble_before": (ev.get("ensemble_now") or {}).get("balanced_acc"),
                              "ensemble_after": (ev.get("ensemble_with_candidate") or {}).get("balanced_acc"),
                              "human_decision": (promo.get("human_decision") or {}).get("decision"),
                              "note": (promo.get("human_decision") or {}).get("note", "")}
    return facts


def collect(question: str = "", min_balanced_acc: float = 0.0, device: str = "cpu") -> tuple[dict, dict]:
    """(facts for the LLM, charts for the platform)."""
    exclusions = load_exclusions()
    zoo = ModelZoo(roots=DEFAULT_MODEL_ROOTS, device=device, exclusions={})     # every run, excluded too
    registry = _registry()

    models, charts_models, probs, y = [], {}, {}, None
    for name, card in zoo.cards.items():
        reason = exclusions.get(name) or exclusions.get(Path(card.path).parent.name) or ""
        if not reason and card.val_balanced_acc < min_balanced_acc:
            reason = f"below the platform's minimum C-val balanced accuracy ({min_balanced_acc})"
        role = "excluded" if reason else "ensemble member"
        val = validation_record(card.path, device)
        p, labels = np.asarray(val["probs"]), np.asarray(val["labels"])
        if y is None:
            y = labels
        try:
            n_params = sum(t.numel() for t in build_from_config(card.config).parameters())
        except Exception:
            n_params = None
        m = _model_facts(card, role, reason, registry, n_params)
        m["c_val"] = _metrics(labels, p)
        models.append(m)
        probs[name] = (p, card.skill, role)
        charts_models[name] = {"role": role, "input_size_px": card.input_size, "roc": _roc(labels, p),
                               "confusion": confusion_matrix(labels, p.argmax(1), labels=list(range(7))).tolist()}

    members = [n for n, (_, _, role) in probs.items() if role == "ensemble member"]
    ensemble = None
    if members:
        w = np.array([probs[n][1] for n in members])
        p_ens = np.tensordot(w / w.sum(), np.stack([probs[n][0] for n in members]), axes=1)
        ensemble = {"members": members, "rule": RULES["soft_vote"],
                    "note": "on C-val every member sees the image at its own input size, so all vote",
                    "c_val": _metrics(y, p_ens)}
        charts_models["ensemble"] = {"role": "ensemble", "roc": _roc(y, p_ens),
                                     "confusion": confusion_matrix(y, p_ens.argmax(1), labels=list(range(7))).tolist()}
    for s in zoo.skipped:                   # runs that could not be scored at all
        models.append({"name": s["name"], "role": "excluded", "excluded_because": s["reason"]})

    c_val_counts = np.bincount(y, minlength=7).tolist() if y is not None else _counts("dermamnist_c", "val")
    facts = {
        "question": question,
        "classes": [{"short": s, "name": n} for s, n in zip(CLASS_SHORT, CLASS_NAMES)],
        "validation_set": {"name": f"{VALIDATION_SET} val", "images": int(sum(c_val_counts or [])),
                           "per_class": dict(zip(CLASS_SHORT, c_val_counts or [])),
                           "description": DATASET_NOTES["dermamnist_c_val"], "why": DATASET_NOTES["why_c_val"]},
        "datasets": {
            "notes": {k: v for k, v in DATASET_NOTES.items() if not k.endswith("c_val") and k != "why_c_val"},
            "class_counts": {
                "dermamnist_official_train": dict(zip(CLASS_SHORT, _counts("dermamnist", "train") or [])),
                "dermamnist_official_val": dict(zip(CLASS_SHORT, _counts("dermamnist", "val") or [])),
                "dermamnist_c_train": dict(zip(CLASS_SHORT, _counts("dermamnist_c", "train") or [])),
                "dermamnist_c_val": dict(zip(CLASS_SHORT, c_val_counts or [])),
                "dermamnist_e_val": dict(zip(CLASS_SHORT, _counts("dermamnist_e", "val") or [])),
            },
        },
        "rules": RULES,
        "chance_balanced_accuracy": _r(CHANCE_BALANCED_ACC),
        "ensemble": ensemble,
        "models": models,
        "exclusions_file": str(EXCLUSIONS_FILE.relative_to(PROJECT_ROOT)),
        "caveats": CAVEATS,
    }
    charts = {"classes": CLASS_SHORT, "validation_set": facts["validation_set"]["name"],
              "per_class_support": facts["validation_set"]["per_class"], "models": charts_models}
    return facts, charts


def summary(facts: dict) -> str:
    """The deterministic answer: what the platform shows without an LLM, or when
    the LLM's answer does not pass the claims check."""
    lines = []
    ens = facts.get("ensemble")
    members = [m for m in facts["models"] if m["role"] == "ensemble member"]
    excluded = [m for m in facts["models"] if m["role"] == "excluded"]
    vs = facts["validation_set"]
    lines.append(f"The ensemble has {len(members)} models, all scored on {vs['name']} "
                 f"({vs['images']} images).")
    for m in sorted(members, key=lambda m: -(m.get("vote_weight") or 0)):
        v = m["c_val"]
        lines.append(f"- {m['name']}: {m['architecture']} at {m['input_size_px']} px, trained on "
                     f"{m['trained_on']}{', ImageNet-pretrained' if m['pretrained_imagenet'] else ''}; "
                     f"balanced accuracy {v['balanced_accuracy']}, macro-F1 {v['macro_f1']}, "
                     f"macro-AUC {v['macro_auc']}, accuracy {v['accuracy']}; vote weight {m['vote_weight']}.")
    if ens:
        v = ens["c_val"]
        lines.append(f"The ensemble's soft vote reaches balanced accuracy {v['balanced_accuracy']}, macro-F1 "
                     f"{v['macro_f1']}, macro-AUC {v['macro_auc']} on the same images (optimistic: the vote "
                     "weights come from them).")
    for m in excluded:
        lines.append(f"- Excluded: {m['name']} — {m.get('excluded_because', '')}")
    lines.append(f"{vs['name']} per class: " + ", ".join(f"{k} {n}" for k, n in vs["per_class"].items()) + ".")
    return "\n".join(lines)
