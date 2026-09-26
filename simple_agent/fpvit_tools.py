#!/usr/bin/env python3
"""
Framework-agnostic wrappers around `train.py`, shaped to be used as agent tools.

Nothing in the existing project is imported or modified: this module launches
`train.py` as a subprocess exactly as a human would from the shell, then reads
the `experiment_record.json` the script writes and compresses it into a small
dict an LLM can actually reason about (a full record contains the whole
per-epoch history with confusion matrices - tens of thousands of tokens).

There is no LangGraph / LangChain / agno import here on purpose: swapping agent
framework means rewriting `agent.py`, not this file.

    from simple_agent.fpvit_tools import train_fpvit, list_runs, get_run_summary

    result = train_fpvit(epochs=3, optimizer="adamw", lr=3e-4, run_name="probe")
    print(result["summary"]["val_at_best"])
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

# --- where things live ------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
RUNS_DIR = PROJECT_ROOT / "runs_agent"      # kept separate from the manual runs/

# --- the choices train.py accepts (mirrored here only to fail fast, before a
#     run is launched: an invalid value should come back as a readable message
#     for the model, not as an argparse traceback after the process starts) ---

OPTIMIZERS = ("sgd", "adamw")
AUG_PRESETS = ("none", "dihedral", "default", "strong", "paper")
CLASS_WEIGHTS = ("none", "inverse", "effective")
SELECT_ON = ("macro_auc", "balanced_acc", "macro_f1", "acc", "loss")

CLASS_NAMES = [
    "actinic keratoses / intraepithelial carcinoma",
    "basal cell carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]
TRAIN_CLASS_COUNTS = [228, 359, 769, 80, 779, 4693, 99]

# One epoch costs ~200 s on MPS, so an unattended agent must never be able to
# start the 5.5 h default run by accident. Both caps are enforced here, and the
# wall-clock one is *also* handed to train.py (--max-seconds), which stops
# cleanly between epochs and still writes every metric it collected.
MAX_EPOCHS = 100
DEFAULT_MAX_SECONDS = 1800.0
HARD_MAX_SECONDS = 6 * 3600.0
# Grace added on top of --max-seconds before this module kills the process
# itself: train.py only checks its budget between epochs.
KILL_GRACE_SECONDS = 900.0


# --- helpers ----------------------------------------------------------------

def _safe_run_name(name: str) -> str:
    """A run name becomes a directory name, so keep it to a harmless alphabet."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "").strip())
    return cleaned or time.strftime("run_%Y%m%d_%H%M%S")


def _thin(seq: list, keep: int = 12) -> list:
    """Evenly sample `keep` items, always keeping the first and the last."""
    if len(seq) <= keep:
        return list(seq)
    step = (len(seq) - 1) / (keep - 1)
    idx = sorted({int(round(i * step)) for i in range(keep)} | {0, len(seq) - 1})
    return [seq[i] for i in idx]


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def summarize_record(record: dict) -> dict:
    """Compress an experiment_record.json into something an LLM can read.

    Keeps: the outcome, the metrics at the selected epoch, per-class recall (the
    only thing that shows whether the rare classes were learned at all), a
    thinned loss curve, and the train/val gap. Drops: confusion matrices,
    per-class precision/F1, and every epoch in between.
    """
    history = record.get("history") or []
    best_epoch = record.get("best_epoch")
    best = next((h for h in history if h.get("epoch") == best_epoch),
                history[-1] if history else {})

    recalls = best.get("val_per_class_recall") or []
    per_class_recall = {
        f"{i} {CLASS_NAMES[i] if i < len(CLASS_NAMES) else ''}".strip(): _round(r, 3)
        for i, r in enumerate(recalls)
    }
    collapsed = [i for i, r in enumerate(recalls) if float(r) == 0.0]

    train_acc, val_acc = best.get("train_acc"), best.get("val_acc")
    gap = None if train_acc is None or val_acc is None else _round(train_acc - val_acc)

    epoch_seconds = [h.get("epoch_seconds", 0.0) for h in history]

    return {
        "stop_reason": record.get("stop_reason"),
        "epochs_requested": (record.get("config") or {}).get("epochs"),
        "epochs_completed": record.get("epochs_completed"),
        "selection_metric": record.get("selection_metric"),
        "best_epoch": best_epoch,
        "best_score": _round(record.get("best_score")),
        "val_at_best": {
            "acc": _round(best.get("val_acc")),
            "macro_auc": _round(best.get("val_macro_auc")),
            "macro_f1": _round(best.get("val_macro_f1")),
            "balanced_acc": _round(best.get("val_balanced_acc")),
            "loss": _round(best.get("val_loss")),
        },
        "per_class_recall_at_best": per_class_recall,
        "collapsed_classes": collapsed,
        "train_val_acc_gap": gap,
        "curve": [
            {
                "epoch": h.get("epoch"),
                "train_loss": _round(h.get("train_loss"), 4),
                "val_loss": _round(h.get("val_loss"), 4),
                "val_balanced_acc": _round(h.get("val_balanced_acc"), 4),
                "val_macro_auc": _round(h.get("val_macro_auc"), 4),
            }
            for h in _thin(history)
        ],
        "mean_epoch_seconds": _round(sum(epoch_seconds) / len(epoch_seconds), 1) if epoch_seconds else None,
        "wallclock_seconds": _round(record.get("wallclock_seconds"), 1),
        "device": record.get("device"),
        "num_params": record.get("num_params"),
        "best_checkpoint": record.get("best_checkpoint"),
        # The knobs that produced this, so a follow-up run can change one thing
        # on purpose instead of guessing what the previous one used.
        "config": {
            k: (record.get("config") or {}).get(k)
            for k in ("epochs", "batch_size", "lr", "weight_decay", "optimizer",
                      "class_weight", "balanced_sampler", "select_on",
                      "aug_preset", "seed", "embed_dim", "depth", "num_heads")
        },
    }


def _read_record(run_dir: Path) -> dict | None:
    path = run_dir / "experiment_record.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# --- the three tools --------------------------------------------------------

def train_fpvit(epochs: int = 3,
                lr: float = 1e-3,
                optimizer: str = "sgd",
                batch_size: int = 128,
                aug_preset: str = "default",
                class_weight: str = "none",
                balanced_sampler: bool = False,
                select_on: str = "macro_auc",
                max_seconds: float = DEFAULT_MAX_SECONDS,
                seed: int = 42,
                run_name: str = "",
                device: str = "auto",
                stream_output: bool = True) -> dict:
    """Train FPViT on DermaMNIST by calling train.py, and summarise the result.

    Returns {"status": "ok"|"error"|"invalid_config", ...}. Errors are returned,
    never raised: an agent has to be able to read the failure and try again.
    """
    # ---- validate before launching anything -------------------------------
    problems: list[str] = []
    if not isinstance(epochs, int) or not (1 <= epochs <= MAX_EPOCHS):
        problems.append(f"epochs must be an integer in 1..{MAX_EPOCHS}, got {epochs!r}")
    if not (1e-6 <= float(lr) <= 1.0):
        problems.append(f"lr must be in 1e-6..1.0, got {lr!r}")
    if optimizer not in OPTIMIZERS:
        problems.append(f"optimizer must be one of {list(OPTIMIZERS)}, got {optimizer!r}")
    if not (8 <= int(batch_size) <= 512):
        problems.append(f"batch_size must be in 8..512, got {batch_size!r}")
    if aug_preset not in AUG_PRESETS:
        problems.append(f"aug_preset must be one of {list(AUG_PRESETS)}, got {aug_preset!r}")
    if class_weight not in CLASS_WEIGHTS:
        problems.append(f"class_weight must be one of {list(CLASS_WEIGHTS)}, got {class_weight!r}")
    if select_on not in SELECT_ON:
        problems.append(f"select_on must be one of {list(SELECT_ON)}, got {select_on!r}")
    if problems:
        return {"status": "invalid_config", "problems": problems,
                "hint": "fix the listed fields and call the tool again; nothing was launched"}

    budget = float(max_seconds) if max_seconds and max_seconds > 0 else DEFAULT_MAX_SECONDS
    budget = min(budget, HARD_MAX_SECONDS)

    run_name = _safe_run_name(run_name)
    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        run_name = f"{run_name}_{time.strftime('%H%M%S')}"
        run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--epochs", str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--optimizer", optimizer,
        "--aug-preset", aug_preset,
        "--class-weight", class_weight,
        "--select-on", select_on,
        "--max-seconds", str(budget),
        "--seed", str(int(seed)),
        "--device", device,
        "--out", str(run_dir),
        "--no-progress",          # tqdm bars are noise in a captured log
        "--log-file", "train.log",  # mirrored into <out>/train.log
    ]
    if balanced_sampler:
        cmd.append("--balanced-sampler")

    # ---- launch ------------------------------------------------------------
    started = time.time()
    deadline = started + budget + KILL_GRACE_SECONDS
    tail: deque[str] = deque(maxlen=40)
    killed = False

    ## ESEGUE IL COMANDO CMD ASSEMBLATO
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        return {"status": "error", "error": f"could not start train.py: {exc}",
                "command": " ".join(cmd)}

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if stream_output:
            print(f"[train] {line}", flush=True)
        if time.time() > deadline and proc.poll() is None:
            # SIGTERM first: train.py installs a handler that flushes the
            # record and the checkpoint before exiting.
            killed = True
            proc.terminate()
    returncode = proc.wait()

    # ---- read back what the run wrote -------------------------------------
    record = _read_record(run_dir)
    elapsed = round(time.time() - started, 1)

    if record is None:
        return {
            "status": "error",
            "error": ("train.py exited without writing experiment_record.json "
                      f"(exit code {returncode})"),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "elapsed_seconds": elapsed,
            "log_tail": list(tail),
            "command": " ".join(cmd),
        }

    result = {
        "status": "ok" if returncode == 0 and not killed else "partial",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "elapsed_seconds": elapsed,
        "exit_code": returncode,
        "summary": summarize_record(record),
        "command": " ".join(cmd),
    }
    if killed:
        result["note"] = ("the wall-clock budget was exceeded and the process was "
                          "stopped; the metrics below cover the completed epochs only")
    if returncode != 0 and not killed:
        result["log_tail"] = list(tail)
    return result


def list_runs() -> dict:
    """List the runs this agent has produced, newest first, with their outcome."""
    if not RUNS_DIR.exists():
        return {"runs_dir": str(RUNS_DIR), "runs": []}

    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        record = _read_record(run_dir)
        if record is None:
            rows.append({"run_name": run_dir.name, "state": "no record (never finished an epoch)"})
            continue
        config = record.get("config") or {}
        rows.append({
            "run_name": run_dir.name,
            "stop_reason": record.get("stop_reason"),
            "epochs_completed": record.get("epochs_completed"),
            "selection_metric": record.get("selection_metric"),
            "best_score": _round(record.get("best_score")),
            "optimizer": config.get("optimizer"),
            "lr": config.get("lr"),
            "class_weight": config.get("class_weight"),
            "aug_preset": config.get("aug_preset"),
        })
    return {"runs_dir": str(RUNS_DIR), "runs": rows}


def get_run_summary(run_name: str) -> dict:
    """Full (compressed) summary of one previous run, by name."""
    run_dir = RUNS_DIR / _safe_run_name(run_name)
    record = _read_record(run_dir)
    if record is None:
        known = [p.name for p in RUNS_DIR.iterdir() if p.is_dir()] if RUNS_DIR.exists() else []
        return {"status": "error",
                "error": f"no experiment_record.json under {run_dir}",
                "known_runs": known}
    return {"status": "ok", "run_name": run_dir.name, "run_dir": str(run_dir),
            "summary": summarize_record(record)}


if __name__ == "__main__":
    # Smoke test without any LLM: python -m simple_agent.fpvit_tools
    print(json.dumps(list_runs(), indent=2))
