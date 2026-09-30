#!/usr/bin/env python3
"""
Isolated test-split evaluation of a frozen checkpoint.

This is the only code that reads the test split. The training and validation
loop never touches it, and this script never trains or tunes anything. Run it
once, on the model you decided to keep, after model selection is finished:
running it during tuning leaks test information into your choices.

The test split is read from the dataset the checkpoint was trained on, at the
checkpoint's own input size and normalisation. `--dataset` points it at
another test split instead - e.g. DermaMNIST-E's (ISIC 2018 test, other
lesions entirely) as an external test for a model trained on DermaMNIST-C.
Never point a model trained on DermaMNIST-E's train split (all of HAM10000)
at DermaMNIST(-C)'s test split: it contains those images.

    python run.py evaluate --checkpoint model/baseline/baseline_paper/best_model.pt
    python run.py evaluate --checkpoint output/runs_224/fpvit_c224_pre_s42/best_model.pt \
        --dataset dermamnist_e

The report is written to output/evaluations/<model name>.json, or
<model name>_<dataset>.json for a test split other than the training one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import paths
from nets.dataset import DATASETS, eval_transform_for, load_split, model_input
from nets.engine import evaluate as evaluate_model, resolve_device
from nets.zoo import build_from_config

NUM_CLASSES = 7


def evaluate_checkpoint(checkpoint: str | Path, device: str = "auto", batch_size: int = 128,
                        num_workers: int = 2, data_root: str | Path = paths.DATASET,
                        out: str | Path | None = None, dataset: str | None = None) -> dict:
    """Evaluate one checkpoint on a test split and write its report. Returns the report.

    `dataset` defaults to the one the checkpoint was trained on."""
    checkpoint = Path(checkpoint)
    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    cfg = ckpt["config"]

    trained_on = cfg.get("dataset") or "dermamnist"
    dataset = dataset or trained_on
    img_size, norm = model_input(cfg)

    # Only the test split is loaded, with the deterministic eval pipeline:
    # no augmentation anywhere, and no train/val data in this process.
    test_set = load_split("test", eval_transform_for(cfg), dataset, img_size, str(data_root))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_from_config(cfg, num_classes=NUM_CLASSES).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Test split: {dataset} ({len(test_set)} images, {img_size}x{img_size}, norm={norm}); "
          f"model trained on {trained_on}")
    m = evaluate_model(model, test_loader, dev, NUM_CLASSES)

    name = checkpoint.parent.name
    report = {
        "model": name, "checkpoint": str(checkpoint), "arch": cfg.get("arch") or "fpvit",
        "trained_on": trained_on, "test_dataset": dataset, "test_images": len(test_set),
        "img_size": img_size,
        "trained_epoch": ckpt.get("epoch"), "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "test_acc": m.acc, "test_macro_auc": m.macro_auc, "test_macro_f1": m.macro_f1,
        "test_balanced_acc": m.balanced_acc,
        "per_class_precision": m.per_class_precision.tolist(),
        "per_class_recall": m.per_class_recall.tolist(),
        "per_class_f1": m.per_class_f1.tolist(),
        "confusion_matrix": m.confusion.tolist(),
    }
    default_name = name if dataset == trained_on else f"{name}_{dataset}"
    out_path = Path(out) if out else paths.EVALUATIONS / f"{default_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(out_path)
    return report


def main():
    p = argparse.ArgumentParser(description="Isolated test-split evaluation of a frozen checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", choices=DATASETS, default=None,
                   help="whose test split to use (default: the dataset the checkpoint was trained on)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=str, default=str(paths.DATASET))
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--out", type=str, default=None,
                   help="report path (default: output/evaluations/<model name>.json)")
    a = p.parse_args()
    report = evaluate_checkpoint(a.checkpoint, a.device, a.batch_size, a.num_workers, a.data_root, a.out,
                                 a.dataset)
    print(json.dumps({k: v for k, v in report.items() if k != "report_path"}, indent=2))
    print(f"\nTest report written to: {report['report_path']}")


if __name__ == "__main__":
    main()
