"""
Facts about a finished segment, computed from its curve. No LLM.

These are what the analyst LLM is given as ground truth and what the training
reviewer checks its diagnosis against: if the LLM says "overfitting" and the
train/val gap did not grow, the facts win and the disagreement is flagged.
"""

from __future__ import annotations

OVERFIT_GAP_GROWTH = 0.05     # train-val accuracy gap growing by this much after the best epoch
VAL_LOSS_RISE = 1.05          # val loss 5 % above its value at the best epoch
UNDERFIT_TRAIN_ACC = 0.80
WEAK_RECALL = 0.2


def segment_facts(s: dict, previous_best: float | None, parent: dict | None) -> dict:
    if s.get("status") == "error":
        return {"status": "error", "flags": ["error"], "error": s.get("error")}
    curve = s["curve"]
    last = curve[-1] if curve else {}
    best = next((c for c in curve if c["epoch"] == s["best_epoch"]), last)
    gap_best = round(best.get("train_acc", 0) - best.get("val_acc", 0), 4)
    gap_last = round(last.get("train_acc", 0) - last.get("val_acc", 0), 4)
    val_loss_rising = bool(best and last.get("val_loss", 0) > best.get("val_loss", 0) * VAL_LOSS_RISE
                           and last.get("train_loss", 0) < best.get("train_loss", 0))
    losses = [c["train_loss"] for c in curve[-3:]]
    train_still_falling = len(losses) >= 2 and losses[-1] < losses[0] * 0.97
    still_improving = s["best_epoch"] == s["epochs_completed"]

    flags = []
    if s.get("stop_reason") == "early_stop":
        flags.append("plateau")
    if gap_last - gap_best > OVERFIT_GAP_GROWTH or val_loss_rising:
        flags.append("overfitting")
    if last.get("train_acc", 1) < UNDERFIT_TRAIN_ACC and train_still_falling:
        flags.append("underfitting")
    if still_improving and s.get("stop_reason") in ("completed", "max_seconds"):
        flags.append("still_improving")
    if s["collapsed_classes"]:
        flags.append("collapsed_classes")
    weak = [c for c, r in s["per_class_recall"].items() if r < WEAK_RECALL]

    score = s.get("best_score")
    return {
        "status": s["status"], "run": s["run"], "arch": s["arch"],
        "stop_reason": s.get("stop_reason"),
        "trigger": {"early_stop": "no improvement for `patience` epochs",
                    "max_seconds": "time budget of the segment spent",
                    "completed": "all epochs run",
                    "interrupted": "stopped by a signal"}.get(s.get("stop_reason"), s.get("stop_reason")),
        "best_epoch": s["best_epoch"], "epochs_completed": s["epochs_completed"],
        "score": score, "selection_metric": s.get("selection_metric"),
        "val_at_best": s["val_at_best"],
        "gap_at_best": gap_best, "gap_at_end": gap_last,
        "val_loss_rising_after_best": val_loss_rising,
        "train_loss_still_falling": train_still_falling,
        "collapsed_classes": s["collapsed_classes"], "weak_classes": weak,
        "per_class_recall": s["per_class_recall"],
        "improvement_vs_campaign_best": (round(score - previous_best, 4)
                                         if score is not None and previous_best is not None else None),
        "improvement_vs_parent": (round(score - parent["best_score"], 4)
                                  if parent and score is not None and parent.get("best_score") is not None
                                  else None),
        "mean_epoch_seconds": s.get("mean_epoch_seconds"),
        "flags": flags,
        "note": "validation has ~1000 images and some classes ~10-20: per-epoch metrics are noisy, "
                "a change below ~0.01 in balanced accuracy is not evidence of anything",
    }
