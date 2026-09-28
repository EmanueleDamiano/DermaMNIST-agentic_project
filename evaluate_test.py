#!/usr/bin/env python3
"""
Run the final, isolated test-set evaluation for a frozen FPViT checkpoint.

This is deliberately a separate script from train.py: the training/validation
loop never touches the test split, and this script never touches training
data or performs any tuning. That separation mirrors WP6 (Testing Agent) in
the AgenticDerma proposal, which requires the test protocol to run without
post-test tuning, on a package that is already frozen.

The test split is read from the dataset the checkpoint was trained on, at the
checkpoint's own input size and normalisation. `--dataset` points it at
another test split instead - e.g. DermaMNIST-E's (ISIC 2018 test, other
lesions entirely) as an external test for a model trained on DermaMNIST-C.
Never point a model trained on DermaMNIST-E's train split (all of HAM10000)
at DermaMNIST(-C)'s test split: it contains those images.

Example:
    python evaluate_test.py --checkpoint runs/fpvit_dermamnist/best_model.pt
    python evaluate_test.py --checkpoint runs_224/fpvit_c224_pre_s42/best_model.pt \
        --dataset dermamnist_e
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torch.utils.data import DataLoader

from fpvit.dataset import DATASETS, eval_transform_for, load_split, model_input
from fpvit.engine import evaluate, resolve_device
from fpvit.zoo import build_from_config

NUM_CLASSES = 7


def parse_args():
    p = argparse.ArgumentParser(description="Isolated test-set evaluation for FPViT")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", choices=DATASETS, default=None,
                    help="whose test split to use (default: the dataset the checkpoint was "
                         "trained on)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | cuda | mps")
    p.add_argument("--out", type=str, default=None,
                    help="where to write the test report JSON (default: test_report_<dataset>.json "
                         "alongside the checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    trained_on = cfg.get("dataset") or "dermamnist"
    dataset = args.dataset or trained_on
    img_size, norm = model_input(cfg)

    # Only the test split is loaded, with the deterministic eval pipeline:
    # no augmentation anywhere, and no train/val data in this process.
    test_set = load_split("test", eval_transform_for(cfg), dataset, img_size, args.data_root)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_from_config(cfg, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Test split: {dataset} ({len(test_set)} images, {img_size}x{img_size}, norm={norm}); "
          f"model trained on {trained_on}")
    test_metrics = evaluate(model, test_loader, device, NUM_CLASSES)

    report = {
        "checkpoint": args.checkpoint,
        "trained_on": trained_on,
        "test_dataset": dataset,
        "test_images": len(test_set),
        "img_size": img_size,
        "trained_epoch": ckpt.get("epoch"),
        "test_acc": test_metrics.acc,
        "test_macro_auc": test_metrics.macro_auc,
        "test_macro_f1": test_metrics.macro_f1,
        "test_balanced_acc": test_metrics.balanced_acc,
        "per_class_precision": test_metrics.per_class_precision.tolist(),
        "per_class_recall": test_metrics.per_class_recall.tolist(),
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "confusion_matrix": test_metrics.confusion.tolist(),
    }

    print(json.dumps(report, indent=2))

    out_path = (Path(args.out) if args.out
                else Path(args.checkpoint).with_name(f"test_report_{dataset}.json"))
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTest report written to: {out_path}")


if __name__ == "__main__":
    main()
