#!/usr/bin/env python3
"""
Run the final, isolated test-set evaluation for a frozen FPViT checkpoint.

This is deliberately a separate script from train.py: the training/validation
loop never touches the test split, and this script never touches training
data or performs any tuning. That separation mirrors WP6 (Testing Agent) in
the AgenticDerma proposal, which requires the test protocol to run without
post-test tuning, on a package that is already frozen.

Example:
    python evaluate_test.py --checkpoint runs/fpvit_dermamnist/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fpvit.dataset import get_dermamnist_loaders
from fpvit.engine import evaluate, resolve_device
from fpvit.model import build_fpvit


def parse_args():
    p = argparse.ArgumentParser(description="Isolated test-set evaluation for FPViT")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | cuda | mps")
    p.add_argument("--out", type=str, default=None,
                    help="where to write the test report JSON (default: alongside the checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # aug=None: no augmentation anywhere. The test pipeline must be
    # deterministic, so this script never builds a train-time transform.
    bundle = get_dermamnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug=None,
        data_root=args.data_root,
    )

    model = build_fpvit(
        num_classes=bundle.num_classes,
        in_channels=bundle.in_channels,
        input_size=28,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        use_resnet_head=not cfg.get("no_resnet_head", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, bundle.test_loader, device, bundle.num_classes)

    report = {
        "checkpoint": args.checkpoint,
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

    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_name("test_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTest report written to: {out_path}")


if __name__ == "__main__":
    main()
