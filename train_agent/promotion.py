"""
Promotion gate: does a candidate make the prediction agent's ensemble better?

    evaluate_candidate()  soft-vote balanced accuracy on the COMMON validation
                          set (DermaMNIST-C val, see predict_agent/validation.py),
                          with the current ensemble and with the candidate
                          added, using the same skill-weighted soft vote as
                          predict_agent.voting. Every model is scored at its
                          own input size (28 or 224 px) on the same 573
                          images; the test split is never loaded.
    promote()             copy the candidate into models_promoted/<new name>/
                          (a fresh folder, never overwritten; the original run
                          stays where it is) and append to the registry. The
                          prediction agent scans models_promoted/, so the model
                          votes from the next prediction on.

Models trained outside the training agent (e.g. on Colab) go through the same
gate from the command line, with the human decision asked on the terminal:

    python -m train_agent.promotion trained_models_224/fpvit_c224_pretrained_s42 [more runs...]

Caveat, stated in every evaluation: the weights of the vote are validation
metrics and the evaluation is on the same validation split, so the numbers are
optimistic in absolute terms. The *comparison* with/without is fair, because
both sides pay the same bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from predict_agent.models import CHANCE_BALANCED_ACC, CLASS_NAMES, PROJECT_ROOT, ModelZoo
from predict_agent.validation import SIDECAR, VALIDATION_SET, validation_record

PROMOTED = PROJECT_ROOT / "models_promoted"
REGISTRY = PROMOTED / "registry.jsonl"
MIN_GAIN = 0.005          # below this the "improvement" is validation noise


def val_probs(ckpt_path: str, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, labels) of one checkpoint on the common validation set."""
    rec = validation_record(ckpt_path, device)
    return np.asarray(rec["probs"], dtype=np.float64), np.asarray(rec["labels"], dtype=np.int64)


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    recall = [float((pred[y == c] == c).mean()) if (y == c).any() else 0.0 for c in range(len(CLASS_NAMES))]
    return {"balanced_acc": round(float(np.mean(recall)), 4), "acc": round(float((pred == y).mean()), 4),
            "per_class_recall": dict(zip(CLASS_NAMES, [round(r, 3) for r in recall]))}


def _soft_vote(members: list[tuple[str, float]], device, y: np.ndarray) -> np.ndarray:
    total = sum(w for _, w in members)
    out = 0.0
    for path, w in members:
        p, labels = val_probs(path, device)
        if not np.array_equal(labels, y):         # 28 and 224 px files must list the same images
            raise ValueError(f"{path}: validation labels differ from the candidate's")
        out = out + w * p
    return out / total


def evaluate_candidate(candidate: dict, device: str = "cpu") -> dict:
    zoo = ModelZoo(device=device)
    cand_p, y = val_probs(candidate["checkpoint"], device)
    cand_bal = _metrics(cand_p.argmax(1), y)["balanced_acc"]
    cand_skill = max(cand_bal - CHANCE_BALANCED_ACC, 0.01)

    current = [(c.path, c.skill) for c in zoo.cards.values()
               if Path(c.path).resolve() != Path(candidate["checkpoint"]).resolve()]
    result = {
        "candidate": candidate["run"], "candidate_alone": _metrics(cand_p.argmax(1), y),
        "candidate_skill": round(cand_skill, 4), "ensemble_size": len(current),
        "ensemble_members": [c.name for c in zoo.cards.values()],
        "excluded": zoo.skipped,
        "validation_set": f"{VALIDATION_SET} val ({len(y)} images)",
        "caveat": "vote weights and evaluation both come from the validation split: absolute "
                  "numbers are optimistic, the with/without comparison is fair",
    }
    if current:
        before = _metrics(_soft_vote(current, device, y).argmax(1), y)
        after = _metrics(_soft_vote(current + [(candidate["checkpoint"], cand_skill)], device, y).argmax(1), y)
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
    # The validation sidecar is keyed by the checkpoint's hash, so the copy
    # stays valid and the zoo does not score the model again.
    for f in ("best_model.pt", "experiment_record.json", "metrics.csv", SIDECAR):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    try:
        source = str(src.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source = str(src.resolve())
    entry = {
        "name": name, "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_run": source, "arch": candidate["arch"],
        "sha256": _sha256(dst / "best_model.pt"), "val_at_best": candidate.get("val_at_best"),
        "validation_set": evaluation.get("validation_set"),
        "ensemble_evaluation": {k: evaluation.get(k) for k in ("ensemble_now", "ensemble_with_candidate",
                                                              "delta_balanced_acc", "improves")},
        "human_decision": decision,
    }
    with open(REGISTRY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


# --------------------------------------------------------------------------
# Command line: the gate for runs trained outside the training agent
# --------------------------------------------------------------------------
def candidate_from_run(run_dir: Path) -> dict:
    """The candidate dict the gate expects, read from a train.py output folder."""
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    return {
        "run": run_dir.name, "dir": str(run_dir), "checkpoint": str(ckpt_path),
        "arch": cfg.get("arch") or "fpvit",
        "selection_metric": ckpt.get("selection_metric"), "best_score": ckpt.get("selection_score"),
        "val_at_best": {"epoch": ckpt.get("epoch"), "trained_on": cfg.get("dataset") or "dermamnist",
                        **{k: ckpt.get("val_metrics", {}).get(k) for k in ("balanced_acc", "macro_auc", "acc")}},
        "hparams": {k: cfg.get(k) for k in ("dataset", "img_size", "pretrained", "optimizer", "lr",
                                            "class_weight", "seed", "epochs")},
    }


def _print_evaluation(ev: dict) -> None:
    now, after = ev["ensemble_now"], ev["ensemble_with_candidate"]
    print(f"  validation:       {ev['validation_set']}")
    print(f"  ensemble now:     {ev['ensemble_members'] or '(empty)'}")
    if ev["excluded"]:
        print(f"  not in ensemble:  {[s['name'] for s in ev['excluded']]}")
    print(f"  candidate alone:  balanced acc {ev['candidate_alone']['balanced_acc']:.4f} "
          f"(vote weight {ev['candidate_skill']:.3f})")
    print(f"  ensemble:         {now['balanced_acc']:.4f} -> {after['balanced_acc']:.4f} "
          f"({ev['delta_balanced_acc']:+.4f}, threshold +{MIN_GAIN})")
    changes = ", ".join(f"{c} {d:+.3f}" for c, d in ev["recall_changes"].items() if d)
    print(f"  recall changes:   {changes or 'none'}")
    print(f"  caveat:           {ev['caveat']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Promotion gate for runs trained outside the training agent")
    p.add_argument("runs", nargs="+", help="train.py output folders (with best_model.pt), gated in order")
    p.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    p.add_argument("--dry-run", action="store_true", help="evaluate only, never promote")
    args = p.parse_args(argv)

    for run in args.runs:
        cand = candidate_from_run(Path(run))
        print(f"\n=== {cand['run']} ===")
        ev = evaluate_candidate(cand, args.device)   # re-evaluated after each promotion
        _print_evaluation(ev)
        print("  recommendation:   " + ("promote" if ev["improves"] else "do not promote"))
        if args.dry_run:
            continue
        if not sys.stdin.isatty():
            print("  no terminal to ask the human: skipped (run it interactively)")
            continue
        answer = input("  Promote into the ensemble? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "s", "si", "sì"):
            print("  not promoted")
            continue
        note = input("  Note for the registry (optional): ").strip()
        entry = promote(cand, ev, "manual", {"decision": "approve", "note": note, "via": "cli"})
        print(f"  promoted as models_promoted/{entry['name']} (sha256 {entry['sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
