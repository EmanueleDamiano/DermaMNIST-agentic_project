"""Inferenza su nuove immagini a partire da un run addestrato.

    python -m training.predict --image lesione.png
    python -m training.predict --image lesione.png --run-id 20260901-171530 --json
    python -m training.predict --dir ./mie_immagini

Il preprocessing merita una nota, perche' non e' quello che verrebbe spontaneo.

Il modello e' stato addestrato su immagini DermaMNIST: 28x28 pixel, poi risalite a 224 per
adattarsi ai pesi ImageNet. Sono quindi immagini **intrinsecamente sfocate**. Una foto
dermatoscopica nativa a 600x450, ridimensionata direttamente a 224, sarebbe molto piu' nitida
di qualsiasi immagine vista in addestramento: fuori distribuzione, con predizioni inaffidabili.

Per questo l'inferenza replica esattamente la catena del training — **prima a 28x28, poi a
224** — anche quando l'immagine di partenza e' ad alta risoluzione. Si butta via del dettaglio
di proposito: e' il prezzo di un modello addestrato su un dataset a bassa risoluzione.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from d4_augmentation import load_image

from .config import CLASS_ABBR, RUNS_DIR, TrainConfig
from .dataset import prepare_batch, resolve_normalization
from .model import build_model, select_device
from .train import list_runs, load_run

# Nomi estesi delle classi, per un output leggibile anche da chi non conosce le sigle.
CLASS_FULL_NAMES = {
    "akiec": "cheratosi attinica / carcinoma intraepiteliale (pre-maligna)",
    "bcc": "carcinoma basocellulare (maligno)",
    "bkl": "lesione cheratosica benigna",
    "df": "dermatofibroma (benigno)",
    "mel": "melanoma (maligno)",
    "nv": "nevo melanocitico (benigno)",
    "vasc": "lesione vascolare (benigna)",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


def find_runs_with_weights() -> list[str]:
    """Elenca i run che hanno i pesi salvati, dal piu' recente."""
    return [r["run_id"] for r in list_runs() if (RUNS_DIR / r["run_id"] / "model.pt").exists()]


def resolve_run_id(run_id: str = "latest") -> str:
    """Risolve 'latest' nel run piu' recente che abbia i pesi salvati."""
    if run_id.strip().lower() in ("latest", "ultimo", "last"):
        available = find_runs_with_weights()
        if not available:
            raise FileNotFoundError(
                "Nessun run con pesi salvati. Addestra un modello con:\n"
                "  python -m training.train --epochs 3 --class-weighting balanced"
            )
        return available[0]

    resolved = run_id.strip()
    if not (RUNS_DIR / resolved / "model.pt").exists():
        available = find_runs_with_weights()
        raise FileNotFoundError(
            f"Il run '{resolved}' non ha pesi salvati (manca model.pt). "
            f"Run utilizzabili: {', '.join(available) if available else 'nessuno'}."
        )
    return resolved


def load_model_from_run(run_id: str = "latest") -> tuple[torch.nn.Module, TrainConfig, torch.device, str]:
    """Ricostruisce il modello di un run e ne carica i pesi.

    L'architettura viene ricostruita dalla config salvata nel run, non dai default correnti:
    un run vecchio resta caricabile anche se i default del progetto cambiano.
    """
    resolved = resolve_run_id(run_id)
    summary = load_run(resolved)
    saved = summary["config"]
    cfg = TrainConfig(**{k: v for k, v in saved.items() if k in TrainConfig.__dataclass_fields__})

    device = select_device()
    # pretrained=False: i pesi ImageNet verrebbero subito sovrascritti da quelli del run.
    model = build_model(pretrained=False, freeze_backbone=False, input_size=cfg.input_size)
    state = torch.load(RUNS_DIR / resolved / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg, device, resolved


def preprocess_image(image: np.ndarray, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    """Porta un'immagine qualsiasi nella stessa distribuzione vista in addestramento.

    Passaggi: RGB -> 28x28 (la risoluzione del dataset) -> input_size del run -> normalizzazione.
    """
    if image.ndim == 2:                      # scala di grigi -> RGB
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:                  # RGBA -> RGB, scarta il canale alfa
        image = image[:, :, :3]
    if image.shape[2] != 3:
        raise ValueError(f"Attesa un'immagine a 1, 3 o 4 canali, ricevuta shape {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Attesa un'immagine uint8, ricevuto dtype {image.dtype}.")

    # copy=True: l'array restituito da PIL e' in sola lettura e torch.from_numpy
    # avvertirebbe di un tensore non scrivibile.
    x = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0)  # (1,H,W,3)

    # Riduzione a 28x28: replica la catena del dataset. `area` e' l'interpolazione corretta
    # per rimpicciolire, evita l'aliasing che `bilinear` introdurrebbe.
    if image.shape[0] != 28 or image.shape[1] != 28:
        small = x.permute(0, 3, 1, 2).float()
        small = F.interpolate(small, size=(28, 28), mode="area")
        x = small.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

    mean_t, std_t = resolve_normalization(cfg)
    mean = torch.tensor(mean_t, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_t, device=device).view(1, 3, 1, 1)
    return prepare_batch(x, cfg.input_size, mean, std, device)


@torch.no_grad()
def classify_array(image: np.ndarray, run_id: str = "latest") -> dict:
    """Classifica un'immagine gia' caricata in memoria."""
    model, cfg, device, resolved = load_model_from_run(run_id)
    x = preprocess_image(image, cfg, device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    order = np.argsort(probs)[::-1]
    return {
        "run_id": resolved,
        "predicted_class": CLASS_ABBR[order[0]],
        "predicted_class_full": CLASS_FULL_NAMES[CLASS_ABBR[order[0]]],
        "confidence": float(probs[order[0]]),
        # FR-C2: si restituisce sempre la distribuzione completa, non solo la top-1.
        "probabilities": {CLASS_ABBR[i]: float(probs[i]) for i in order},
    }


def classify_file(path: str | Path, run_id: str = "latest") -> dict:
    """Classifica un'immagine su disco. Solleva FileNotFoundError/ValueError se non valida."""
    path = Path(path).expanduser()
    if path.is_dir():
        found = sorted(p.name for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        raise ValueError(
            f"'{path}' e' una directory, non un file. "
            + (f"Immagini trovate: {', '.join(found[:20])}." if found else "Non contiene immagini.")
        )
    image = load_image(path)          # riusa il loader di d4_augmentation.py
    result = classify_array(image, run_id)
    result["image"] = str(path)
    return result


def list_images(directory: str | Path) -> list[str]:
    """Elenca i file immagine di una directory, in ordine alfabetico."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"Directory non trovata: {directory}")
    if not directory.is_dir():
        raise ValueError(f"'{directory}' non e' una directory.")
    return sorted(str(p) for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def format_result(result: dict) -> str:
    """Rende il risultato in testo leggibile."""
    lines = [
        f"Immagine: {result.get('image', '(in memoria)')}",
        f"Classe predetta: {result['predicted_class']} — {result['predicted_class_full']}",
        f"Confidenza: {result['confidence']:.1%}",
        "Distribuzione completa delle probabilita':",
    ]
    lines += [f"  {k:>5}: {v:6.2%}" for k, v in result["probabilities"].items()]
    lines.append(f"Modello: run {result['run_id']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.predict",
        description="Classifica una lesione cutanea con un modello gia' addestrato.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Percorso dell'immagine da classificare.")
    group.add_argument("--dir", help="Classifica tutte le immagini di una directory.")
    group.add_argument("--list-runs", action="store_true", help="Elenca i run con pesi salvati.")
    parser.add_argument("--run-id", default="latest", help="Run da usare (default: il piu' recente).")
    parser.add_argument("--json", action="store_true", help="Output in JSON.")
    args = parser.parse_args(argv)

    if args.list_runs:
        runs = find_runs_with_weights()
        print(json.dumps(runs, indent=2) if runs else "Nessun run con pesi salvati.")
        return 0

    try:
        if args.dir:
            paths = list_images(args.dir)
            if not paths:
                print(f"Nessuna immagine in {args.dir}.")
                return 1
            results = [classify_file(p, args.run_id) for p in paths]
            if args.json:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                for r in results:
                    print(format_result(r))
                    print("-" * 50)
        else:
            result = classify_file(args.image, args.run_id)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else format_result(result))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Errore: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
