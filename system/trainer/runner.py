"""
Launch one training segment of train.py and report what happened.

Framework-free. Two properties are enforced here, whatever the agent asks:

*   **Nothing is ever overwritten.** Every run gets a fresh directory
    output/campaigns/<campaign>/<run>/; if it exists, the run is refused. A warm
    restart reads its parent's checkpoint and writes to a NEW directory
    (train.py --init-from refuses to write into its source as well).
    output/campaigns/ is not scanned by the prediction agent, so candidates never
    reach the ensemble until a human promotes them (promotion.py).
*   **The wall-clock budget belongs to the runner.** It is passed to
    train.py --max-seconds (clean stop between epochs) and enforced again
    here with SIGTERM + grace, which train.py handles by saving.

The segment's own stopping trigger is train.py's early stop: when the
selection metric has not improved for `patience` epochs the segment ends with
stop_reason "early_stop", and the agent wakes up to decide what to change.
"""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from tester.models import CLASS_NAMES, PROJECT_ROOT
from trainer.space import resolved_augmentation

import paths

TRAIN_SCRIPT = paths.SYSTEM / "train.py"
RUNS_TRAIN = paths.CAMPAIGNS
KILL_GRACE_SECONDS = 300.0


def build_argv(p: dict, plan: dict, out_dir: Path, max_seconds: float,
               init_from: str | None) -> list[str]:
    hp = p["hparams"]
    argv = [sys.executable, str(TRAIN_SCRIPT),
            "--arch", p["arch"],
            "--epochs", str(int(hp["epochs"])),
            "--batch-size", str(int(hp["batch_size"])),
            "--lr", repr(float(hp["lr"])),
            "--weight-decay", repr(float(hp["weight_decay"])),
            "--optimizer", hp["optimizer"],
            "--class-weight", hp["class_weight"],
            "--cb-beta", repr(float(hp["cb_beta"])),
            "--select-on", plan["select_on"],
            "--early-stop-patience", str(int(plan["patience"])),
            "--max-seconds", str(int(max_seconds)),
            "--aug-config", json.dumps(resolved_augmentation(p["augmentation"])),
            "--seed", str(int(plan.get("seed", 42))),
            "--num-workers", str(int(plan.get("num_workers", 2))),
            "--out", str(out_dir),
            "--no-progress", "--log-file"]
    if p["arch"] == "fpvit":
        argv += ["--embed-dim", str(int(hp["embed_dim"])), "--depth", str(int(hp["depth"])),
                 "--num-heads", str(int(hp["num_heads"]))]
    if hp.get("balanced_sampler"):
        argv.append("--balanced-sampler")
    if init_from:
        argv += ["--init-from", init_from]
    return argv


def _read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    out = []
    for r in rows:
        try:
            out.append({k: (float(v) if k != "epoch" else int(v)) for k, v in r.items() if v not in (None, "")})
        except ValueError:
            continue                                   # a row being written right now
    return out


def run_segment(p: dict, plan: dict, campaign_dir: Path, run_name: str, max_seconds: float,
                init_from: str | None = None,
                on_epoch: Callable[[dict], None] | None = None) -> dict:
    """Train one segment; return a trial summary (see summarize)."""
    out_dir = campaign_dir / run_name
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists: runs are never overwritten")
    out_dir.mkdir(parents=True)
    argv = build_argv(p, plan, out_dir, max_seconds, init_from)
    (out_dir / "agent_command.json").write_text(json.dumps({"argv": argv, "proposal": p}, indent=1))

    started = time.time()
    seen = 0
    proc = subprocess.Popen(argv, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    stderr_tail: list[str] = []
    killed = False
    try:
        while proc.poll() is None:
            time.sleep(2.0)
            rows = _read_metrics(out_dir / "metrics.csv")
            for row in rows[seen:]:
                if on_epoch:
                    on_epoch({"run": run_name, **row})
            seen = max(seen, len(rows))
            if time.time() - started > max_seconds + KILL_GRACE_SECONDS and not killed:
                proc.send_signal(signal.SIGTERM)       # train.py saves on SIGTERM
                killed = True
        _, err = proc.communicate(timeout=60)
        stderr_tail = (err or "").strip().splitlines()[-15:]
    except BaseException:
        proc.kill()
        raise
    for row in _read_metrics(out_dir / "metrics.csv")[seen:]:
        if on_epoch:
            on_epoch({"run": run_name, **row})

    record_path = out_dir / "experiment_record.json"
    if proc.returncode != 0 or not record_path.exists():
        return {"run": run_name, "status": "error", "arch": p["arch"], "hparams": p["hparams"],
                "augmentation": p["augmentation"], "dir": str(out_dir), "checkpoint": None,
                "error": "\n".join(stderr_tail) or f"train.py exited with {proc.returncode}",
                "seconds": round(time.time() - started, 1), "init_from": init_from}
    record = json.loads(record_path.read_text())
    return summarize(record, p, run_name, out_dir, init_from, time.time() - started, killed)


def _thin(seq: list, keep: int = 15) -> list:
    if len(seq) <= keep:
        return seq
    step = (len(seq) - 1) / (keep - 1)
    return [seq[round(i * step)] for i in range(keep)]


def summarize(record: dict, p: dict, run_name: str, out_dir: Path, init_from: str | None,
              seconds: float, killed: bool) -> dict:
    """The evidence an agent needs, a few hundred tokens instead of a 400 KB record."""
    hist = record.get("history", [])
    best_epoch = record.get("best_epoch")
    best = next((h for h in hist if h["epoch"] == best_epoch), hist[-1] if hist else {})
    recall = [round(x, 3) for x in best.get("val_per_class_recall", [])]
    curve = [{"epoch": h["epoch"], "lr": h.get("lr"),
              "train_loss": round(h["train_loss"], 4), "val_loss": round(h["val_loss"], 4),
              "train_acc": round(h["train_acc"], 4), "val_acc": round(h["val_acc"], 4),
              "val_balanced_acc": round(h["val_balanced_acc"], 4),
              "val_macro_f1": round(h["val_macro_f1"], 4)} for h in hist]
    ckpt = out_dir / "best_model.pt"
    return {
        "run": run_name, "status": "partial" if killed else "ok", "arch": p["arch"],
        "hparams": p["hparams"], "augmentation": p["augmentation"],
        "dir": str(out_dir), "checkpoint": str(ckpt) if ckpt.exists() else None,
        "init_from": init_from,
        "selection_metric": record.get("selection_metric"),
        "best_score": record.get("best_score"), "best_epoch": best_epoch,
        "epochs_completed": record.get("epochs_completed"),
        "stop_reason": record.get("stop_reason"),
        "val_at_best": {k: round(best.get(f"val_{k}", 0.0), 4)
                        for k in ("acc", "balanced_acc", "macro_f1", "macro_auc")},
        "per_class_recall": dict(zip(CLASS_NAMES, recall)),
        "collapsed_classes": [CLASS_NAMES[i] for i, r in enumerate(recall) if r == 0.0],
        "curve": _thin(curve),
        "curve_full_length": len(curve),
        "seconds": round(seconds, 1),
        "mean_epoch_seconds": round(sum(h.get("epoch_seconds", 0) for h in hist) / max(len(hist), 1), 1),
    }
