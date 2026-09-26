#!/usr/bin/env python3
"""
Train FPViT on DermaMNIST.

Example:
    python train.py --epochs 100 --batch-size 128 --lr 1e-3 --out runs/fpvit_run1

This script is written to fit the "Training Agent" contract described in the
AgenticDerma project proposal (WP3): it takes a fixed data split, a fixed
experiment policy (config below), trains with a stored random seed, and
writes a versioned experiment record (config + metrics + checkpoint path) to
`<out>/experiment_record.json` so the run can be reproduced and compared by
the Evaluation Agent later. It does not access the test split.

Monitoring artefacts written into `<out>`, all refreshed *every epoch* so a
long run that is interrupted keeps everything logged up to that point:

    experiment_record.json  full record: config, seed, per-epoch history
                            (including per-class precision/recall/F1 and the
                            confusion matrix), best epoch, stop reason,
                            resume state
    metrics.csv             one flat row per epoch, for plotting / `tail -f`
    best_model.pt           weights at the best value of --select-on
    last_model.pt           weights + optimizer/scheduler/RNG state, so an
                            interrupted run can continue with `--resume`
    train.log               optional copy of stdout, with `--log-file`

and one line appended to a shared index (default `<out>/../index.jsonl`)
when the run ends, so a tuning agent can read the whole campaign's history
without reopening every per-run record.

Three knobs exist for automated search specifically:

    --select-on             which validation metric picks `best_model.pt`.
                            macro-AUC (the default, and the paper's headline
                            metric) is weakly sensitive to imbalance: it is
                            possible for the selected checkpoint to have a
                            worse balanced accuracy than a later one.
    --early-stop-patience   stop once that metric has stopped improving
    --max-seconds           stop once a wall-clock budget is spent, so a
                            scheduler (successive halving / ASHA) can hand
                            every configuration a fixed slice of time

Because DermaMNIST is heavily imbalanced (67 % melanocytic nevi), the
per-class metrics are the ones that actually tell you whether the rare
classes are being learned; plain accuracy sits near the majority-class
baseline of 0.669 even for a model that has collapsed onto one class.
`--class-weight` and `--balanced-sampler` are the two levers against that
imbalance; both are deviations from the paper's recipe, so they default to
off and are recorded in the experiment record when used.

Augmentation is selected in one of three ways, which compose (later wins):

    --aug-preset none|dihedral|default|strong|paper
    --aug-config '{"cj_hue": 0.01, ...}'   (inline JSON, or a path to a file,
                                            or a previous experiment_record.json)
    --aug-set cj_hue=0.01 --aug-set cutout=true      (repeatable)

The resolved config is stored field by field under "augmentation" in
experiment_record.json and in both checkpoints, so a run is repeatable by
feeding that file back through --aug-config. See fpvit/augment.py for the
search space a tuning agent should sample from.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fpvit.augment import PRESETS, parse_overrides, resolve_augmentation
from fpvit.dataset import get_dermamnist_loaders
from fpvit.engine import (compute_class_weights, evaluate, resolve_device,
                           train_one_epoch)
from fpvit.zoo import ARCHITECTURES, DEFAULT_ARCH, build_model


# Flat, scalar-only columns for metrics.csv. Per-class arrays and the
# confusion matrix go to experiment_record.json instead, where they don't
# explode into 20+ columns.
CSV_FIELDS = [
    "epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
    "val_macro_auc", "val_macro_f1", "val_balanced_acc", "epoch_seconds",
]


# Which validation metric selects best_model.pt, and whether bigger is better.
# Maps the --select-on name to (EpochMetrics attribute, history/CSV column,
# direction). Keeping the history column here means the best epoch can be
# recomputed from a stored record alone, which is what --resume does when the
# selection metric has changed between invocations.
SELECTION_METRICS = {
    "macro_auc":    ("macro_auc",    "val_macro_auc",    "max"),
    "balanced_acc": ("balanced_acc", "val_balanced_acc", "max"),
    "macro_f1":     ("macro_f1",     "val_macro_f1",     "max"),
    "acc":          ("acc",          "val_acc",          "max"),
    "loss":         ("loss",         "val_loss",         "min"),
}


def worst_score(key: str) -> float:
    """The sentinel that any real score beats."""
    return -math.inf if SELECTION_METRICS[key][2] == "max" else math.inf


def selection_score(val_metrics, key: str) -> float:
    """Reads the selected metric off an EpochMetrics.

    macro_auc is None when a class is absent from the validation split, so it
    falls back to accuracy - the same fallback the original macro-AUC-only
    selection used.
    """
    value = getattr(val_metrics, SELECTION_METRICS[key][0])
    return val_metrics.acc if value is None else value


def history_score(entry: dict, key: str):
    value = entry.get(SELECTION_METRICS[key][1])
    return entry.get("val_acc") if value is None else value


def is_better(new: float, reference: float, key: str, min_delta: float = 0.0) -> bool:
    """min_delta is the margin an epoch must clear to count as an improvement.

    Used with 0.0 to decide whether to checkpoint, and with
    --early-stop-min-delta to decide whether to reset the patience counter -
    so a run is not kept alive by improvements in the fifth decimal.
    """
    if SELECTION_METRICS[key][2] == "max":
        return new > reference + min_delta
    return new < reference - min_delta


def best_from_history(history: list[dict], key: str) -> tuple[float, int | None]:
    """Recomputes (best score, best epoch) from a stored history.

    Needed on --resume: the caller may have changed --select-on, and comparing
    a fresh macro-F1 against a stored macro-AUC would silently keep the wrong
    checkpoint forever.
    """
    best, best_epoch = worst_score(key), None
    for entry in history:
        score = history_score(entry, key)
        if score is None:
            continue
        if is_better(score, best, key):
            best, best_epoch = score, entry["epoch"]
    return best, best_epoch


def append_index_entry(path: Path, entry: dict):
    """Appends one JSON line to the campaign index, locking around the write.

    Several runs may be in flight at once under a search scheduler, and an
    entry carrying the resolved augmentation is larger than the write size
    POSIX guarantees to be atomic, so the append takes an exclusive lock.
    Failing to write the index must never take down a finished training run,
    so every error here is reported and swallowed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with open(path, "a") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass  # no flock (non-POSIX, or a filesystem without it)
            f.write(line)
            f.flush()
    except OSError as exc:
        print(f"WARNING: could not append to index {path}: {exc}")


def install_sigterm_handler():
    """Turns SIGTERM into KeyboardInterrupt.

    A scheduler that stops an unpromising trial sends SIGTERM; routing it
    through the same path as Ctrl-C means the run still writes its final
    record and its index line instead of vanishing.
    """
    def handler(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not on the main thread
        pass


class Tee:
    """Mirrors stdout into a log file.

    tqdm writes to stderr, so the progress bars stay out of the log file and
    it remains greppable even without `--no-progress`.
    """

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = open(path, "a", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # set_rng_state insists on a CPU ByteTensor, so undo any map_location that
    # pushed the saved state onto the accelerator.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["torch_cuda"]])


def write_json_atomic(path: Path, payload: dict):
    """Writes via a temp file + rename, so an interrupt mid-write cannot
    leave a truncated record behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def write_metrics_csv(path: Path, history: list[dict]):
    """Rewrites the whole CSV from the history each epoch.

    A full rewrite (rather than an append) keeps the file consistent with
    the history after a `--resume`, and 100 rows is nothing.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def build_parser() -> argparse.ArgumentParser:
    """Split out from parse_args so a caller can read the defaults.

    `fpvit_runner` needs them to tell "the same configuration, resumed" from
    "a different configuration reusing the folder name", without keeping a
    second copy of every default that would drift from this one.
    """
    p = argparse.ArgumentParser(description="Train FPViT on DermaMNIST")
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), default=DEFAULT_ARCH,
                    help="network to train (default: fpvit). The CNNs ignore the FPViT-only "
                         "--embed-dim/--depth/--num-heads/--no-resnet-head")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd",
                    help="paper uses SGD with lr=1e-3; adamw is a stronger default for ViT heads")
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=4, help="Transformer layers per ViT head (paper default: 4)")
    p.add_argument("--num-heads", type=int, default=3, help="attention heads per ViT head")
    p.add_argument("--no-resnet-head", action="store_true",
                    help="drop the 4th (ResNet) head -> reproduces the paper's '3 heads' ablation")

    # --- class imbalance (both are deviations from the paper's recipe) ------
    p.add_argument("--class-weight", choices=["none", "inverse", "effective"], default="none",
                    help="per-class weights in the TRAINING loss, from the train-split counts "
                         "[228,359,769,80,779,4693,99]. 'inverse' = N/(K*n_c); 'effective' = "
                         "Cui et al. 2019, softened by --cb-beta. Validation loss stays unweighted")
    p.add_argument("--cb-beta", type=float, default=0.999,
                    help="beta for --class-weight effective; closer to 1 is closer to inverse frequency")
    p.add_argument("--balanced-sampler", action="store_true",
                    help="resample the train split so each epoch is class-balanced in expectation; "
                         "an alternative to --class-weight, not a complement")

    # --- model selection and stopping --------------------------------------
    p.add_argument("--select-on", choices=sorted(SELECTION_METRICS), default="macro_auc",
                    help="validation metric that picks best_model.pt (default: macro_auc, the "
                         "paper's metric, but weakly sensitive to class imbalance)")
    p.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop after this many epochs without improvement in --select-on "
                         "(0 = disabled, i.e. always run all --epochs)")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="improvement below this margin does not reset the patience counter")
    p.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-clock budget for THIS invocation (0 = unlimited). The loop stops "
                         "before starting an epoch it predicts would exceed the budget, so a "
                         "search scheduler can give each trial a fixed slice of time")
    p.add_argument("--aug-preset", choices=sorted(PRESETS), default="default",
                    help="named augmentation preset (see fpvit/augment.py PRESETS)")
    p.add_argument("--aug-config", type=str, default=None,
                    help="augmentation as inline JSON or a path to a JSON file; "
                         "replaces --aug-preset entirely. An experiment_record.json "
                         "can be passed directly (its 'augmentation' key is used)")
    p.add_argument("--aug-set", action="append", default=None, metavar="KEY=VALUE",
                    help="override one augmentation field, repeatable "
                         "(e.g. --aug-set cj_hue=0.01 --aug-set cutout=true)")
    p.add_argument("--no-cutout", action="store_true",
                    help="convenience override forcing cutout=false after resolution")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", type=str, default=None,
                    help="folder containing dermamnist.npz if it should not be auto-downloaded")
    p.add_argument("--out", type=str, default="runs/fpvit_dermamnist")
    p.add_argument("--index-file", type=str, default=None,
                    help="JSONL campaign index, one line appended per finished run "
                         "(default: <out>/../index.jsonl; 'none' disables it)")
    p.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | cuda | mps (auto picks cuda, then mps, then cpu)")
    p.add_argument("--resume", action="store_true",
                    help="continue from <out>/last_model.pt (model, optimizer, scheduler, RNG, history)")
    p.add_argument("--init-from", type=str, default=None, metavar="CHECKPOINT",
                    help="warm start: load only the WEIGHTS of this checkpoint (best_model.pt or "
                         "last_model.pt of an earlier run of the same architecture), then train "
                         "with a fresh optimizer and schedule - so --lr/--weight-decay take effect. "
                         "Writes to --out, never to the source run. Not combinable with --resume")
    p.add_argument("--no-progress", action="store_true",
                    help="disable tqdm progress bars (use for nohup / CI logs)")
    p.add_argument("--log-file", type=str, nargs="?", const="train.log", default=None,
                    help="mirror stdout to this file; bare flag defaults to <out>/train.log")
    return p


def parse_args():
    return build_parser().parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tee = None
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute() and log_path.parent == Path("."):
            log_path = out_dir / log_path
        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee

    try:
        run(args, out_dir)
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


def run(args, out_dir: Path):
    set_seed(args.seed)
    install_sigterm_handler()
    device = resolve_device(args.device)
    show_progress = not args.no_progress
    run_started_at = time.time()

    print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # One resolution path shared with programmatic callers, so a config an
    # agent builds in Python and one passed on the CLI behave identically.
    overrides = parse_overrides(args.aug_set)
    if args.no_cutout:
        overrides["cutout"] = False
    aug = resolve_augmentation(args.aug_preset, args.aug_config, overrides)

    print("Augmentation (train split only):")
    for line in aug.describe():
        print(f"  - {line}")
    print()

    bundle = get_dermamnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug=aug,
        data_root=args.data_root,
        balanced_sampler=args.balanced_sampler,
    )

    # Two ways to correct the same imbalance; applying both corrects it twice
    # and over-represents the rare classes. Allowed, because an ablation may
    # want exactly that, but never silently.
    if args.balanced_sampler and args.class_weight != "none":
        print("WARNING: --balanced-sampler and --class-weight are both active; "
              "the class imbalance is being corrected twice.\n")

    class_weights = compute_class_weights(bundle.class_counts, args.class_weight, args.cb_beta)
    print(f"Train class counts: {bundle.class_counts}")
    if args.balanced_sampler:
        print("Sampling:           balanced (with replacement)")
    if class_weights is not None:
        print(f"Loss weights ({args.class_weight}): "
              + " ".join(f"{i}:{w:.2f}" for i, w in enumerate(class_weights.tolist())))
    print()

    model = build_model(
        arch=args.arch,
        num_classes=bundle.num_classes,
        in_channels=bundle.in_channels,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        use_resnet_head=not args.no_resnet_head,
    ).to(device)
    print(f"Architecture: {args.arch} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    if args.init_from:
        # Weights only: the optimizer, scheduler and history start fresh, which is
        # the point - a warm restart with a different lr / weight decay.
        if args.resume:
            raise ValueError("--init-from and --resume are mutually exclusive")
        src = Path(args.init_from)
        if src.resolve().parent == out_dir.resolve():
            raise ValueError("--init-from must point to a different run than --out "
                             "(a warm restart never overwrites its source)")
        state = torch.load(src, map_location=device, weights_only=False)
        src_cfg = state.get("config", {})
        src_arch = src_cfg.get("arch") or DEFAULT_ARCH
        if src_arch != args.arch:
            raise ValueError(f"--init-from checkpoint is {src_arch!r}, but --arch is {args.arch!r}")
        if args.arch == "fpvit":
            shape = {k: src_cfg.get(k) for k in ("embed_dim", "depth", "num_heads", "no_resnet_head")}
            wanted = {"embed_dim": args.embed_dim, "depth": args.depth,
                      "num_heads": args.num_heads, "no_resnet_head": args.no_resnet_head}
            if any(shape[k] is not None and shape[k] != wanted[k] for k in wanted):
                raise ValueError(f"--init-from checkpoint has FPViT shape {shape}, run asks for {wanted}")
        model.load_state_dict(state["model_state_dict"])
        print(f"Warm start from: {src} (epoch {state.get('epoch')}, "
              f"{state.get('selection_metric')}={state.get('selection_score')})\n")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_ckpt_path = out_dir / "best_model.pt"
    last_ckpt_path = out_dir / "last_model.pt"
    record_path = out_dir / "experiment_record.json"
    csv_path = out_dir / "metrics.csv"

    best_score = worst_score(args.select_on)
    best_epoch = None
    patience_ref = worst_score(args.select_on)
    epochs_without_improvement = 0
    stop_reason = "completed"
    history: list[dict] = []
    start_epoch = 1

    if args.resume:
        if not last_ckpt_path.exists():
            raise FileNotFoundError(
                f"--resume given but {last_ckpt_path} does not exist. "
                "Only a run that already wrote last_model.pt can be resumed."
            )
        state = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng_state(state.get("rng_state"))
        history = state.get("history", [])
        start_epoch = state["epoch"] + 1

        # Recompute the best epoch from the stored history under the CURRENT
        # --select-on rather than trusting the stored scalar, which may have
        # been measured with a different metric.
        stored_best_epoch = state.get("best_epoch")
        best_score, best_epoch = best_from_history(history, args.select_on)
        patience_ref = best_score
        epochs_without_improvement = state.get("epochs_without_improvement", 0)
        if best_epoch is not None and stored_best_epoch is not None and best_epoch != stored_best_epoch:
            print(f"  WARNING: under --select-on {args.select_on} the best epoch of the "
                  f"resumed history is {best_epoch}, not {stored_best_epoch}. "
                  f"{best_ckpt_path.name} still holds the weights of epoch {stored_best_epoch} "
                  f"and will only be overwritten once a new epoch beats "
                  f"{best_score:.4f}.")
        score_str = f"{best_score:.4f}" if best_epoch is not None else "n/a"
        print(f"Resumed from {last_ckpt_path}: next epoch is {start_epoch}, "
              f"best val {args.select_on} so far {score_str} (epoch {best_epoch})")

        # The scheduler's state_dict carries T_max, so resuming restores the
        # ORIGINAL cosine schedule. Changing --epochs on a resume therefore
        # does not stretch the schedule: with a larger value the lr stays at
        # eta_min for the extra epochs. Warn instead of failing, since the
        # user may well want exactly that.
        prev_cfg = state.get("config", {})
        changed = {k: (prev_cfg.get(k), v) for k, v in vars(args).items()
                   if k not in {"resume", "log_file", "no_progress", "device",
                                "num_workers", "out", "data_root",
                                "max_seconds", "index_file"}
                   and k in prev_cfg and prev_cfg.get(k) != v}
        if changed:
            print("  WARNING: config differs from the resumed run:")
            for k, (was, now) in changed.items():
                print(f"    {k}: {was} -> {now}")
            if "epochs" in changed:
                print(f"    the cosine schedule keeps its original T_max="
                      f"{prev_cfg.get('epochs')}; the lr will NOT be re-stretched "
                      f"over {args.epochs} epochs.")

        # Compare the RESOLVED augmentation too: the preset name can be
        # identical while the fields differ (a --aug-set override, or a
        # changed preset definition), which would silently make the resumed
        # epochs incomparable to the earlier ones.
        prev_aug = state.get("augmentation")
        if prev_aug is not None and prev_aug != aug.to_dict():
            diff = {k: (prev_aug.get(k), v) for k, v in aug.to_dict().items()
                    if prev_aug.get(k) != v}
            print("  WARNING: augmentation differs from the resumed run:")
            for k, (was, now) in diff.items():
                print(f"    {k}: {was} -> {now}")
        if start_epoch > args.epochs:
            print(f"Nothing to do: --epochs {args.epochs} already reached.")
            return

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, {bundle.num_classes} classes")
    print(f"Data:  train {len(bundle.train_loader.dataset)} / "
          f"val {len(bundle.val_loader.dataset)} / test {len(bundle.test_loader.dataset)} (untouched)")
    print(f"Epochs {start_epoch}..{args.epochs}\n")

    def best_epoch_entry() -> dict:
        """The history row of the currently selected epoch ({} if there is none)."""
        return next((h for h in history if h["epoch"] == best_epoch), {})

    def save_record():
        write_json_atomic(record_path, {
            "config": vars(args),
            # The RESOLVED augmentation, not just the preset name: a preset's
            # contents can change between versions, so the record stores every
            # field explicitly. Feed this file back with --aug-config to repeat
            # the exact pipeline.
            "augmentation": aug.to_dict(),
            "augmentation_ops": aug.describe(),
            "seed": args.seed,
            "num_params": n_params,
            "device": str(device),
            "selection_metric": args.select_on,
            # worst_score() is +/-inf, which is not valid JSON, so a run that
            # has not completed an epoch reports no score at all.
            "best_score": None if best_epoch is None else best_score,
            # Kept under its original name so existing readers keep working,
            # but it is now the macro-AUC *of the selected epoch*, which is
            # only the best macro-AUC when --select-on is macro_auc.
            "best_val_macro_auc": best_epoch_entry().get("val_macro_auc"),
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
            "epochs_completed": history[-1]["epoch"] if history else 0,
            # "completed" | "early_stop" | "max_seconds" | "interrupted".
            # A search scheduler needs this to tell a converged trial from a
            # truncated one before comparing their scores.
            "stop_reason": stop_reason,
            "train_class_counts": bundle.class_counts,
            "class_weights": None if class_weights is None else class_weights.tolist(),
            "wallclock_seconds": time.time() - run_started_at,
            "history": history,
            "num_classes": bundle.num_classes,
            "task": bundle.task,
        })
        write_metrics_csv(csv_path, history)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Wall-clock budget, checked *before* the epoch: stopping after
            # the epoch that overruns would make the budget an average rather
            # than a cap. The first epoch of an invocation is never blocked,
            # otherwise a budget smaller than one epoch would train nothing
            # and report no score at all.
            if args.max_seconds > 0 and history and epoch > start_epoch:
                elapsed = time.time() - run_started_at
                recent = [h["epoch_seconds"] for h in history[-5:]]
                predicted = sum(recent) / len(recent)
                if elapsed + predicted > args.max_seconds:
                    stop_reason = "max_seconds"
                    print(f"\nWall-clock budget reached: {elapsed:.0f}s of "
                          f"{args.max_seconds:.0f}s used, and the next epoch needs "
                          f"~{predicted:.0f}s more. Stopping after epoch {epoch - 1}.")
                    break

            t0 = time.time()
            # Read the lr *before* scheduler.step() so the logged value is the one
            # the epoch actually trained with.
            epoch_lr = optimizer.param_groups[0]["lr"]

            train_metrics = train_one_epoch(model, bundle.train_loader, optimizer, device,
                                             epoch, progress=show_progress,
                                             class_weights=class_weights)
            val_metrics = evaluate(model, bundle.val_loader, device, bundle.num_classes,
                                    progress=show_progress)
            scheduler.step()
            dt = time.time() - t0

            history.append({
                "epoch": epoch,
                "lr": epoch_lr,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.acc,
                "val_loss": val_metrics.loss,
                "val_acc": val_metrics.acc,
                "val_macro_auc": val_metrics.macro_auc,
                "val_macro_f1": val_metrics.macro_f1,
                "val_balanced_acc": val_metrics.balanced_acc,
                "epoch_seconds": dt,
                "val_per_class_precision": val_metrics.per_class_precision.tolist(),
                "val_per_class_recall": val_metrics.per_class_recall.tolist(),
                "val_per_class_f1": val_metrics.per_class_f1.tolist(),
                "val_confusion_matrix": val_metrics.confusion.tolist(),
            })

            # ETA from the mean of the last few epochs, which is steadier than the
            # last one alone.
            recent = [h["epoch_seconds"] for h in history[-5:]]
            eta = (args.epochs - epoch) * (sum(recent) / len(recent))

            score = selection_score(val_metrics, args.select_on)
            is_best = is_better(score, best_score, args.select_on)
            if is_best:
                best_score = score
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_metric": args.select_on,
                    "selection_score": score,
                    "val_metrics": {
                        "acc": val_metrics.acc,
                        "macro_auc": val_metrics.macro_auc,
                        "macro_f1": val_metrics.macro_f1,
                        "balanced_acc": val_metrics.balanced_acc,
                        "per_class_precision": val_metrics.per_class_precision.tolist(),
                        "per_class_recall": val_metrics.per_class_recall.tolist(),
                        "per_class_f1": val_metrics.per_class_f1.tolist(),
                        "confusion_matrix": val_metrics.confusion.tolist(),
                    },
                    "config": vars(args),
                    "augmentation": aug.to_dict(),
                }, best_ckpt_path)

            # Patience keeps its own reference: with --early-stop-min-delta an
            # epoch can be the new best (so worth checkpointing) while still
            # being too small an improvement to buy the run more epochs.
            if is_better(score, patience_ref, args.select_on, args.early_stop_min_delta):
                patience_ref = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"[epoch {epoch:03d}/{args.epochs}] lr={epoch_lr:.2e} "
                  f"train: {train_metrics.summary()} | val: {val_metrics.summary()} "
                  f"({dt:.1f}s, eta {format_eta(eta)}){'  * best' if is_best else ''}")
            # Recall per class is the number that exposes a collapse onto the
            # majority class, which plain accuracy hides on this dataset.
            print("            val recall/class: " +
                  " ".join(f"{i}:{r:.2f}" for i, r in enumerate(val_metrics.per_class_recall)))

            # Written every epoch: an interrupted 5-hour run keeps its history.
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": rng_state(),
                "epoch": epoch,
                "selection_metric": args.select_on,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "config": vars(args),
                "augmentation": aug.to_dict(),
            }, last_ckpt_path)
            save_record()

            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                stop_reason = "early_stop"
                print(f"\nEarly stop: val {args.select_on} has not improved by more than "
                      f"{args.early_stop_min_delta} for {epochs_without_improvement} epochs "
                      f"(best {best_score:.4f} at epoch {best_epoch}).")
                break
    except KeyboardInterrupt:
        # Also reached on SIGTERM, i.e. when a search scheduler kills an
        # unpromising trial: the record and the index entry still get written.
        stop_reason = "interrupted"
        print("\nInterrupted: keeping everything written so far.")

    save_record()

    # Final test-set evaluation is intentionally NOT run here: per the WP4/WP6
    # split of responsibilities in the project proposal, the frozen model is
    # evaluated on the test split only inside the isolated Testing Agent, after
    # model selection is complete. Use evaluate_test.py for that step.
    score_str = f"{best_score:.4f}" if best_epoch is not None else "n/a"
    print(f"\nDone ({stop_reason}). Best val {args.select_on}: {score_str} (epoch {best_epoch})")
    print(f"Best checkpoint:    {best_ckpt_path}")
    print(f"Last checkpoint:    {last_ckpt_path}")
    print(f"Experiment record:  {record_path}")
    print(f"Per-epoch metrics:  {csv_path}")

    # One line per finished run, appended to a shared index. This is the file
    # a tuning agent reads to remember the campaign: the per-run records hold
    # the full per-epoch history and are far too big to keep re-reading.
    if args.index_file != "none":
        index_path = Path(args.index_file) if args.index_file else out_dir.parent / "index.jsonl"
        best = best_epoch_entry()
        append_index_entry(index_path, {
            "run_id": out_dir.name,
            "out_dir": str(out_dir),
            "record": str(record_path),
            "checkpoint": str(best_ckpt_path),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stop_reason": stop_reason,
            "device": str(device),
            "seed": args.seed,
            "num_params": n_params,
            "epochs_requested": args.epochs,
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "wallclock_seconds": round(time.time() - run_started_at, 1),
            "mean_epoch_seconds": (round(sum(h["epoch_seconds"] for h in history) / len(history), 1)
                                   if history else None),
            "selection_metric": args.select_on,
            "best_score": None if best_epoch is None else best_score,
            "best_epoch": best_epoch,
            "hparams": {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "optimizer": args.optimizer,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "no_resnet_head": args.no_resnet_head,
                "class_weight": args.class_weight,
                "cb_beta": args.cb_beta,
                "balanced_sampler": args.balanced_sampler,
            },
            "augmentation": aug.to_dict(),
            # Metrics at the SELECTED epoch. Per-class recall is included
            # because it is the only field that shows whether the long tail
            # was learned at all - the aggregate metrics hide it.
            "best_metrics": {k: best.get(k) for k in (
                "train_loss", "train_acc", "val_loss", "val_acc",
                "val_macro_auc", "val_macro_f1", "val_balanced_acc",
                "val_per_class_recall",
            )},
        })
        print(f"Campaign index:     {index_path}")


if __name__ == "__main__":
    main()
