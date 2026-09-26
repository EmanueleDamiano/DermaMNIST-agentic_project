"""Esecuzione di un esperimento di training, con CLI propria.

Questo modulo e' una funzione pura sul piano concettuale: `train_once(cfg)` prende una
configurazione e produce una run directory con config, metriche e log. Non sa nulla
dell'agente e non lo importa. E' cio' che permette di riprodurre da riga di comando
qualsiasi esperimento che l'agente abbia lanciato:

    python -m training.train --epochs 3 --augmentation d4 --class-weighting balanced
    python -m training.train --list
    python -m training.train --show <run_id>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .config import RUNS_DIR, TrainConfig
from .dataset import class_weights_tensor, make_loaders, prepare_batch, resolve_normalization
from .metrics import compare_with_baseline, compute_metrics, format_metrics
from .model import build_model, count_parameters, select_device

ProgressCallback = Callable[[str], None]


def set_seed(seed: int) -> None:
    """Fissa i seed. Nota: su MPS alcune operazioni non sono bit-esatte fra run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class _RunLogger:
    """Scrive su file e, opzionalmente, inoltra a un callback (usato per lo streaming)."""

    def __init__(self, path: Path, callback: ProgressCallback | None = None):
        self.path = path
        self.callback = callback
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        with open(self.path, "a") as f:
            f.write(line + "\n")
        if self.callback:
            self.callback(line)


@torch.no_grad()
def evaluate(model, loader, cfg, mean, std, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restituisce (y_true, y_pred, y_proba) su tutto il loader."""
    model.eval()
    trues, probas = [], []
    for images, labels in loader:
        x = prepare_batch(images, cfg.input_size, mean, std, device)
        logits = model(x)
        probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        trues.append(labels.numpy())
    y_true = np.concatenate(trues)
    y_proba = np.concatenate(probas)
    return y_true, y_proba.argmax(axis=1), y_proba


def train_once(cfg: TrainConfig, progress: ProgressCallback | None = None) -> dict:
    """Esegue un esperimento completo e restituisce il riepilogo del run.

    Il modello selezionato e' quello con la migliore balanced accuracy di validation, non
    quello dell'ultima epoca: con classi rare la metrica oscilla molto fra epoche.
    """
    cfg.validate()
    set_seed(cfg.seed)

    run_id = _new_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _RunLogger(run_dir / "training.log", progress)

    device = select_device()
    mean_t, std_t = resolve_normalization(cfg)
    mean = torch.tensor(mean_t, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=device).view(1, 3, 1, 1)

    log(f"run {run_id} | device={device.type}")
    log(f"config: {json.dumps(cfg.to_dict())}")

    loaders = make_loaders(cfg)
    model = build_model(
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        input_size=cfg.input_size,
    ).to(device)
    trainable, total = count_parameters(model)
    log(f"ResNet-18 | parametri addestrabili {trainable:,} su {total:,}")

    weights = class_weights_tensor(cfg, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
    )

    best_val = -1.0
    best_state = None
    best_epoch = -1
    history = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, n_batches = 0.0, 0

        for batch_idx, (images, labels) in enumerate(loaders["train"]):
            if cfg.max_train_batches is not None and batch_idx >= cfg.max_train_batches:
                break
            x = prepare_batch(images, cfg.input_size, mean, std, device)
            y = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        y_true, y_pred, y_proba = evaluate(model, loaders["val"], cfg, mean, std, device)
        val_metrics = compute_metrics(y_true, y_pred, y_proba)
        elapsed = time.time() - epoch_start

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "seconds": elapsed}
        )
        log(
            f"epoca {epoch}/{cfg.epochs} | loss={train_loss:.4f} | "
            f"val {format_metrics(val_metrics)} | {elapsed:.1f}s"
        )

        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"modello selezionato: epoca {best_epoch} (val bal_acc={best_val:.4f})")

    y_true, y_pred, y_proba = evaluate(model, loaders["test"], cfg, mean, std, device)
    test_metrics = compute_metrics(y_true, y_pred, y_proba)
    log(f"test {format_metrics(test_metrics)}")

    summary = {
        "run_id": run_id,
        "config": cfg.to_dict(),
        "device": device.type,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val,
        "history": history,
        "test": test_metrics,
        "baseline_comparison": compare_with_baseline(test_metrics),
        "total_seconds": round(time.time() - started, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    if cfg.save_weights and best_state is not None:
        torch.save(best_state, run_dir / "model.pt")

    log(f"run completato in {summary['total_seconds']}s -> {run_dir}")
    return summary


def list_runs() -> list[dict]:
    """Elenca i run presenti, dal piu' recente. Ignora le directory incomplete."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for path in sorted(RUNS_DIR.iterdir(), reverse=True):
        metrics_file = path / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        runs.append(
            {
                "run_id": data["run_id"],
                "epochs": data["config"]["epochs"],
                "augmentation": data["config"]["augmentation"],
                "class_weighting": data["config"]["class_weighting"],
                "test_balanced_accuracy": round(data["test"]["balanced_accuracy"], 4),
                "test_auc_ovr": round(data["test"]["auc_ovr"], 4),
                "seconds": data["total_seconds"],
            }
        )
    return runs


def load_run(run_id: str) -> dict:
    """Carica il riepilogo completo di un run. Solleva FileNotFoundError se non esiste."""
    metrics_file = RUNS_DIR / run_id / "metrics.json"
    if not metrics_file.exists():
        available = [r["run_id"] for r in list_runs()][:5]
        raise FileNotFoundError(
            f"Run '{run_id}' non trovato. Run disponibili (piu' recenti): "
            f"{', '.join(available) if available else 'nessuno'}."
        )
    with open(metrics_file) as f:
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.train",
        description="Addestra una ResNet-18 pre-addestrata su DermaMNIST.",
    )
    defaults = TrainConfig()
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--input-size", type=int, default=defaults.input_size)
    p.add_argument("--augmentation", choices=["none", "d4"], default=defaults.augmentation)
    p.add_argument(
        "--class-weighting", choices=["none", "balanced"], default=defaults.class_weighting
    )
    p.add_argument("--normalization", choices=["imagenet", "dataset"], default=defaults.normalization)
    p.add_argument("--no-pretrained", action="store_true", help="Addestra da zero.")
    p.add_argument("--freeze-backbone", action="store_true", help="Linear probe: addestra solo la testa.")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--max-train-batches", type=int, default=None, help="Tronca l'epoca (smoke test).")
    p.add_argument(
        "--no-save-weights",
        action="store_true",
        help="Non salvare i pesi (~45 MB). Senza pesi il run non e' usabile per classificare.",
    )
    p.add_argument("--notes", default="")
    p.add_argument("--list", action="store_true", help="Elenca i run esistenti ed esce.")
    p.add_argument("--show", metavar="RUN_ID", help="Mostra il riepilogo di un run ed esce.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        runs = list_runs()
        print(json.dumps(runs, indent=2) if runs else "Nessun run presente.")
        return 0

    if args.show:
        try:
            print(json.dumps(load_run(args.show), indent=2))
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        return 0

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        input_size=args.input_size,
        augmentation=args.augmentation,
        class_weighting=args.class_weighting,
        normalization=args.normalization,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        save_weights=not args.no_save_weights,
        notes=args.notes,
    )
    try:
        summary = train_once(cfg, progress=print)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Errore: {exc}")
        return 1

    print(f"\nrun_id: {summary['run_id']}")
    print(json.dumps(summary["baseline_comparison"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
