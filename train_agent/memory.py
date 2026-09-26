"""
The training agent's deterministic memory. Exact reads of files on disk, no
ranking, no guessing:

    past_runs()        every experiment_record.json the project has, in any
                       runs folder (campaigns, simple_agent, manual runs),
                       compacted to what a proposer needs
    lessons()          from runs_train/campaigns.jsonl: what earlier campaigns
                       changed and what it did to the score
    ensemble_state()   the models the prediction agent votes with today, and
                       which classes none of them handles well - a concrete
                       reason to train something new
    log_trial()        append this campaign's trial to campaigns.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from predict_agent.models import CLASS_NAMES, DEFAULT_MODEL_ROOTS, PROJECT_ROOT
from train_agent.runner import RUNS_TRAIN

CAMPAIGN_LOG = RUNS_TRAIN / "campaigns.jsonl"
WEAK_RECALL = 0.3


def _compact_record(path: Path) -> dict | None:
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cfg = rec.get("config", {})
    hist = rec.get("history", [])
    best = next((h for h in hist if h.get("epoch") == rec.get("best_epoch")), hist[-1] if hist else {})
    recall = best.get("val_per_class_recall", [])
    return {
        "run": path.parent.name, "dir": str(path.parent.relative_to(PROJECT_ROOT)),
        "arch": cfg.get("arch") or "fpvit", "optimizer": cfg.get("optimizer"), "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"), "epochs": cfg.get("epochs"),
        "class_weight": cfg.get("class_weight"), "balanced_sampler": cfg.get("balanced_sampler"),
        "init_from": cfg.get("init_from"),
        "selection_metric": rec.get("selection_metric"),
        "best_score": round(rec["best_score"], 4) if rec.get("best_score") is not None else None,
        "val_balanced_acc": round(best.get("val_balanced_acc", 0.0), 4),
        "epochs_completed": rec.get("epochs_completed"), "stop_reason": rec.get("stop_reason"),
        "collapsed_classes": [CLASS_NAMES[i] for i, r in enumerate(recall) if r == 0.0],
        "mtime": path.stat().st_mtime,
    }


def past_runs(limit: int = 20) -> list[dict]:
    roots = [r for r in DEFAULT_MODEL_ROOTS if r.name != "models_promoted"]
    paths = [p for r in roots if r.is_dir() for p in r.glob("*/experiment_record.json")]
    if RUNS_TRAIN.is_dir():
        paths += list(RUNS_TRAIN.glob("*/*/experiment_record.json"))
    runs = [r for r in (_compact_record(p) for p in paths) if r]
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    for r in runs:
        r.pop("mtime")
    return runs[:limit]


def lessons(limit: int = 15) -> list[dict]:
    """(change -> effect) pairs from earlier campaigns, newest first."""
    if not CAMPAIGN_LOG.exists():
        return []
    out = []
    for line in CAMPAIGN_LOG.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "trial" and e.get("changes") and e.get("delta") is not None:
            out.append({k: e.get(k) for k in ("campaign", "run", "arch", "action", "changes",
                                              "diagnosis_before", "delta", "score")})
    return out[::-1][:limit]


def log_trial(entry: dict) -> None:
    CAMPAIGN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMPAIGN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry},
                           ensure_ascii=False, default=str) + "\n")


def ensemble_state() -> dict:
    from predict_agent.models import ModelZoo
    zoo = ModelZoo(device="cpu")
    models = [{k: c.public()[k] for k in ("name", "arch", "epoch", "val_balanced_acc", "blind_classes")}
              for c in zoo.cards.values()]
    best_recall = [max((c.val_per_class_recall[i] for c in zoo.cards.values()), default=0.0)
                   for i in range(len(CLASS_NAMES))]
    return {
        "models": models,
        "best_val_balanced_acc": max((m["val_balanced_acc"] for m in models), default=0.0),
        "best_recall_per_class": dict(zip(CLASS_NAMES, [round(r, 3) for r in best_recall])),
        "weak_classes": [CLASS_NAMES[i] for i, r in enumerate(best_recall) if r < WEAK_RECALL],
    }
