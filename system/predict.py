#!/usr/bin/env python3
"""
Single-image (or folder) inference for a FPViT checkpoint.

This is the smallest end-to-end path through the pipeline: load an image,
apply exactly the same eval-time preprocessing used by `nets/dataset.py`
(no augmentation), run one forward pass, print the class probabilities.

It is meant for two things:
  1. a smoke test of the whole stack without downloading DermaMNIST
     (run it with no --checkpoint: the weights are random, so only the
     *mechanics* are being verified, never the prediction);
  2. inspecting what a trained checkpoint says about one specific lesion,
     which is the per-case explanation the AgenticDerma Evaluation Agent
     (WP4) needs to attach to a single patient image.

Examples:
    # mechanics only, untrained weights
    python run.py predict --image input/samples/05_melanoma.png

    # real prediction from a trained checkpoint
    python run.py predict --image input/samples/05_melanoma.png \
        --checkpoint model/baseline/baseline_paper/best_model.pt

    # whole folder, scored against input/samples/labels.csv
    python run.py predict --image input/samples --labels input/samples/labels.csv \
        --checkpoint model/baseline/baseline_paper/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from nets.dataset import DATA_FLAG, eval_transform_for, model_input
from nets.engine import resolve_device
from nets.model import build_fpvit
from nets.zoo import build_from_config

from medmnist import INFO

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="FPViT inference on one image or a folder")
    p.add_argument("--image", type=str, required=True,
                   help="path to an image file, or to a folder of images")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="trained checkpoint from train.py; omit to run with random "
                        "weights (mechanics smoke test only)")
    p.add_argument("--labels", type=str, default=None,
                   help="optional CSV with columns file,class_id[,expected_class] "
                        "to score the predictions against")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--out", type=str, default=None,
                   help="optional path to write the predictions as JSON")
    # Only used when no checkpoint is given (a checkpoint carries its own config).
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--no-resnet-head", action="store_true")
    return p.parse_args()


def collect_images(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not target.exists():
        raise FileNotFoundError(f"No such image or folder: {target}")
    return [target]


def load_ground_truth(labels_path: Path) -> dict[str, int]:
    # utf-8-sig: the shipped labels.csv starts with a BOM.
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        return {row["file"]: int(row["class_id"]) for row in csv.DictReader(f)}


def main():
    args = parse_args()
    device = resolve_device(args.device)

    info = INFO[DATA_FLAG]
    # medmnist labels are keyed by the class index as a string: {"0": "...", ...}
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    num_classes = len(class_names)
    in_channels = info["n_channels"]

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = build_from_config(cfg, num_classes=num_classes, in_channels=in_channels).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        provenance = f"checkpoint={args.checkpoint} (epoch {ckpt.get('epoch')})"
        trained = True
    else:
        model = build_fpvit(
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=28,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            use_resnet_head=not args.no_resnet_head,
        ).to(device)
        provenance = "RANDOM (untrained) weights"
        trained = False
        cfg = {}

    model.eval()
    # Resizes any input to the model's own size (28, or 224 for a DermaMNIST-C
    # model) and applies the normalisation it was trained with.
    transform = eval_transform_for(cfg)
    img_size, norm = model_input(cfg)

    images = collect_images(Path(args.image))
    ground_truth = load_ground_truth(Path(args.labels)) if args.labels else {}

    print(f"Device:  {device}")
    print(f"Weights: {provenance}")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input:   {img_size}x{img_size}, norm={norm}")
    if not trained:
        print("\n!! No checkpoint given: the predictions below are meaningless.")
        print("!! This run only verifies that the pipeline executes end to end.")
    print()

    results, n_correct, n_scored = [], 0, 0

    for path in images:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        k = min(args.topk, num_classes)
        top_p, top_i = probs.topk(k)
        pred = int(top_i[0])

        record = {
            "file": path.name,
            "predicted_class_id": pred,
            "predicted_class": class_names[pred],
            "confidence": float(top_p[0]),
            "probabilities": {class_names[i]: float(probs[i]) for i in range(num_classes)},
        }

        print(f"--- {path.name} ---")
        truth = ground_truth.get(path.name)
        if truth is not None:
            n_scored += 1
            hit = truth == pred
            n_correct += int(hit)
            record["true_class_id"] = truth
            record["true_class"] = class_names[truth]
            record["correct"] = hit
            print(f"  ground truth: [{truth}] {class_names[truth]}")
        for prob, idx in zip(top_p.tolist(), top_i.tolist()):
            mark = "<-- pred" if idx == pred else ""
            print(f"  {prob:6.2%}  [{idx}] {class_names[idx]} {mark}")
        if truth is not None:
            print(f"  => {'CORRECT' if truth == pred else 'WRONG'}")
        print()

        results.append(record)

    if n_scored:
        print(f"Scored {n_correct}/{n_scored} correct "
              f"({n_correct / n_scored:.1%}) on {len(images)} image(s).")
        if not trained:
            print("(random weights: chance level is ~1/7 = 14.3%)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"weights": provenance, "trained": trained,
                       "predictions": results}, f, indent=2)
        print(f"\nPredictions written to: {out_path}")


if __name__ == "__main__":
    main()
