"""
The training agent's action space, and the rules that decide what is allowed.

The LLM proposes; this module disposes. Everything here is deterministic and
framework-free:

    normalize_proposal   fill defaults, coerce types, drop unknown keys
    validate_proposal    hard errors (the proposal goes back to the proposer)
                         and the effective train.py configuration
    diff_configs         what a proposal changes with respect to a reference run
    gate                 does this proposal need a human, under the autonomy
                         level? ("guarded" = small lr / weight-decay moves on the
                         same architecture pass alone, everything else asks)

The augmentation rules encode what the dermoscopy modality allows, the same
reasons documented in nets/augment.py: the dihedral group is free, colour is
diagnostic, borders must not be filled with black, aspect changes deform the
lesion.
"""

from __future__ import annotations

import math
from typing import Any

from nets.augment import AUG_SEARCH_SPACE, PRESETS, resolve_augmentation
from nets.zoo import ARCHITECTURES, DEFAULT_ARCH

ACTIONS = ("new_run", "warm_restart", "stop")
OPTIMIZERS = ("adamw", "sgd")
CLASS_WEIGHTS = ("none", "inverse", "effective")
SELECT_ON = ("balanced_acc", "macro_f1", "macro_auc")     # never plain accuracy: 67 % is one class
AUTONOMY = ("supervised", "guarded", "autonomous")

# Bounds a proposal must respect. Guard rails against a typo or a
# hallucinated value, not a search space.
BOUNDS = {
    "lr": (1e-6, 0.5),
    "weight_decay": (0.0, 0.5),
    "batch_size": (16, 512),
    "epochs": (1, 200),
    "cb_beta": (0.9, 0.99999),
    "embed_dim": (48, 384),
    "depth": (1, 8),
    "num_heads": (1, 8),
}
# lr sanity per optimizer: AdamW above 1e-2 diverges on these nets; SGD needs more.
LR_MAX = {"adamw": 1e-2, "sgd": 0.5}

# "guarded": what may pass without a human.
GUARDED_FIELDS = {"lr", "weight_decay", "epochs"}
GUARDED_LR_RATIO = (0.3, 3.0)
GUARDED_WD_RATIO = (0.1, 10.0)

# Colour is diagnostic in dermoscopy: beyond these the pigment cues that
# separate melanoma, nevi and keratoses start to be erased.
AUG_COLOUR_WARN = {"cj_hue": 0.03, "cj_saturation": 0.2}

DEFAULT_HPARAMS = {
    "optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05, "batch_size": 128,
    "epochs": 30, "class_weight": "effective", "cb_beta": 0.999,
    "balanced_sampler": False,
    "embed_dim": 192, "depth": 4, "num_heads": 3,
}
# A reasonable first lr per architecture with AdamW (the CNNs tolerate more).
DEFAULT_LR = {"fpvit": 3e-4, "resnet18": 1e-3, "efficientnet_b0": 1e-3, "convnext_tiny": 4e-4}


def normalize_proposal(raw: dict, reference: dict | None = None) -> dict:
    """A complete proposal dict. Missing fields inherit from `reference`
    (the parent or the previous trial), then from the defaults."""
    raw = dict(raw or {})
    ref = reference or {}
    ref_h = ref.get("hparams", {})
    arch = raw.get("arch") or ref.get("arch") or DEFAULT_ARCH
    hp = {**DEFAULT_HPARAMS, "lr": DEFAULT_LR.get(arch, 3e-4), **ref_h, **(raw.get("hparams") or {})}
    aug_raw = raw.get("augmentation") or ref.get("augmentation") or {"preset": "default", "overrides": {}}
    return {
        "action": raw.get("action") or "new_run",
        "arch": arch,
        "parent_run": raw.get("parent_run") or None,
        "hparams": {k: hp[k] for k in DEFAULT_HPARAMS},
        "augmentation": {"preset": aug_raw.get("preset") or "default",
                         "overrides": dict(aug_raw.get("overrides") or {})},
        "augmentation_rationale": (raw.get("augmentation_rationale") or "").strip(),
        "rationale": (raw.get("rationale") or "").strip(),
        "addresses": [str(a) for a in (raw.get("addresses") or [])],
        "expected_effect": (raw.get("expected_effect") or "").strip(),
        "knowledge_used": [str(c) for c in (raw.get("knowledge_used") or [])],
    }


def _num(value, cast=float):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return None
    return v if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else None


def resolved_augmentation(aug: dict) -> dict:
    """The field-by-field augmentation train.py will use (raises ValueError if invalid)."""
    return resolve_augmentation(aug["preset"], None, dict(aug.get("overrides") or {})).to_dict()


def validate_proposal(p: dict, plan: dict, runs: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(errors, warnings). Any error sends the proposal back to the proposer.

    `runs` maps run names of this campaign to their trial summaries.
    """
    errors, warnings = [], []
    hp = p["hparams"]

    if p["action"] not in ACTIONS:
        errors.append(f"action must be one of {ACTIONS}")
        return errors, warnings
    if p["action"] == "stop":
        return errors, warnings

    if p["arch"] not in ARCHITECTURES:
        errors.append(f"arch must be one of {sorted(ARCHITECTURES)}")
    if plan.get("arch_fixed") and p["arch"] != plan["arch"]:
        errors.append(f"the human fixed the architecture to {plan['arch']!r}")

    if hp["optimizer"] not in OPTIMIZERS:
        errors.append(f"optimizer must be one of {OPTIMIZERS}")
    if hp["class_weight"] not in CLASS_WEIGHTS:
        errors.append(f"class_weight must be one of {CLASS_WEIGHTS}")
    for key, (lo, hi) in BOUNDS.items():
        cast = int if key in ("batch_size", "epochs", "embed_dim", "depth", "num_heads") else float
        v = _num(hp.get(key), cast)
        if v is None or not lo <= v <= hi:
            errors.append(f"{key}={hp.get(key)!r} outside [{lo}, {hi}]")
        else:
            hp[key] = v
    if not errors and hp["lr"] > LR_MAX.get(hp["optimizer"], 1.0):
        errors.append(f"lr={hp['lr']} is too high for {hp['optimizer']} (max {LR_MAX[hp['optimizer']]})")
    if hp["epochs"] > plan["max_epochs_per_run"]:
        errors.append(f"epochs={hp['epochs']} exceeds the plan's {plan['max_epochs_per_run']} per run")
    if p["arch"] == "fpvit" and hp["embed_dim"] % hp["num_heads"]:
        errors.append("embed_dim must be divisible by num_heads")
    if bool(hp["balanced_sampler"]) and hp["class_weight"] != "none":
        errors.append("class_weight and balanced_sampler are alternatives: using both corrects the "
                      "imbalance twice (pick one)")

    # --- augmentation: valid, and coherent with dermoscopy ----------------------
    aug = p["augmentation"]
    if aug["preset"] not in PRESETS:
        errors.append(f"augmentation preset must be one of {sorted(PRESETS)}")
    unknown = set(aug["overrides"]) - set(AUG_SEARCH_SPACE)
    if unknown:
        errors.append(f"unknown augmentation fields {sorted(unknown)}")
    resolved = None
    if not errors:
        try:
            resolved = resolved_augmentation(aug)
        except (ValueError, TypeError) as exc:
            errors.append(f"augmentation invalid: {exc}")
    if resolved:
        for field, limit in AUG_COLOUR_WARN.items():
            if resolved.get("color_jitter") and float(resolved.get(field, 0)) > limit:
                warnings.append(f"{field}={resolved[field]} above {limit}: colour is diagnostic in "
                                "dermoscopy, strong colour jitter can erase pigment cues")
        if float(resolved.get("rotation_degrees", 0)) > 0:
            warnings.append("arbitrary-angle rotation interpolates and fills corners on 28x28 images; "
                            "rotate_90 already gives exact rotations")
        if not p["augmentation_rationale"]:
            errors.append("augmentation_rationale is required: explain why this augmentation suits "
                          "dermoscopic images and this run's diagnosis")

    # --- warm restart: a parent of the same shape, in this campaign ------------
    if p["action"] == "warm_restart":
        parent = runs.get(p["parent_run"] or "")
        if parent is None:
            errors.append(f"warm_restart needs parent_run = one of this campaign's runs {sorted(runs)}")
        elif not parent.get("checkpoint"):
            errors.append(f"parent run {p['parent_run']} has no checkpoint to start from")
        else:
            if parent["arch"] != p["arch"]:
                errors.append(f"warm_restart keeps the architecture: parent is {parent['arch']}")
            if p["arch"] == "fpvit":
                for k in ("embed_dim", "depth", "num_heads"):
                    if parent["hparams"].get(k) != hp[k]:
                        errors.append(f"warm_restart keeps the network shape: {k} differs from the parent")

    # --- not a repeat -------------------------------------------------------------
    sig = config_signature(p)
    for name, r in runs.items():
        if r.get("signature") == sig:
            errors.append(f"this exact configuration was already run as {name}; change something or stop")
            break
    return errors, warnings


def config_signature(p: dict) -> str:
    hp = p["hparams"]
    keys = ["optimizer", "lr", "weight_decay", "batch_size", "epochs", "class_weight",
            "balanced_sampler"] + (["embed_dim", "depth", "num_heads"] if p["arch"] == "fpvit" else [])
    if hp.get("class_weight") == "effective":
        keys.append("cb_beta")
    try:
        aug = resolved_augmentation(p["augmentation"])
    except (ValueError, TypeError):
        aug = p["augmentation"]
    return repr((p["action"], p["arch"], p.get("parent_run") if p["action"] == "warm_restart" else None,
                 tuple((k, round(float(hp[k]), 10) if isinstance(hp[k], float) else hp[k]) for k in keys),
                 tuple(sorted(aug.items()))))


def diff_configs(p: dict, ref: dict | None) -> dict[str, tuple[Any, Any]]:
    """Fields that differ between a proposal and a reference trial."""
    if not ref:
        return {}
    out = {}
    if p["arch"] != ref["arch"]:
        out["arch"] = (ref["arch"], p["arch"])
    for k, v in p["hparams"].items():
        if k in ("embed_dim", "depth", "num_heads") and p["arch"] != "fpvit":
            continue
        if ref["hparams"].get(k) != v:
            out[k] = (ref["hparams"].get(k), v)
    try:
        a, b = resolved_augmentation(ref["augmentation"]), resolved_augmentation(p["augmentation"])
        changed = {k: (a.get(k), b.get(k)) for k in b if a.get(k) != b.get(k)}
    except (ValueError, TypeError, KeyError):
        changed = {"augmentation": (ref.get("augmentation"), p["augmentation"])}
    if changed:
        out["augmentation"] = changed
    return out


def gate(p: dict, ref: dict | None, autonomy: str, first: bool,
         findings: list[dict]) -> tuple[bool, list[str]]:
    """(needs_human, reasons). The autonomy policy, in one auditable function."""
    if p["action"] == "stop":
        return False, []
    serious = [f for f in findings if f["severity"] in ("critical", "warning")]
    if autonomy == "supervised":
        return True, ["autonomy 'supervised': every proposal is approved by a human"]
    if serious:
        return True, [f"the training reviewer flagged: {f['description']}" for f in serious]
    if autonomy == "autonomous":
        return False, []

    # guarded
    reasons = []
    if first or ref is None:
        reasons.append("first run of the campaign")
    changes = diff_configs(p, ref)
    outside = sorted(set(changes) - GUARDED_FIELDS)
    if outside:
        reasons.append("changes outside the automatic envelope: " + ", ".join(outside))
    if "lr" in changes and ref:
        r = p["hparams"]["lr"] / max(ref["hparams"]["lr"], 1e-12)
        if not GUARDED_LR_RATIO[0] - 1e-9 <= r <= GUARDED_LR_RATIO[1] + 1e-9:   # float slack
            reasons.append(f"lr changes by x{r:.2f} (automatic only within x{GUARDED_LR_RATIO[0]}–x{GUARDED_LR_RATIO[1]})")
    if "weight_decay" in changes and ref:
        old, new = ref["hparams"]["weight_decay"], p["hparams"]["weight_decay"]
        r = new / old if old else math.inf
        if not GUARDED_WD_RATIO[0] - 1e-9 <= r <= GUARDED_WD_RATIO[1] + 1e-9:
            reasons.append(f"weight_decay {old} → {new} outside the automatic range")
    return bool(reasons), reasons
