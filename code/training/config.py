"""Percorsi e configurazione condivisi dal layer di training.

La configurazione e' un dataclass congelato: viene serializzata in ogni run directory, cosi'
un esperimento e' sempre ricostruibile dai suoi artefatti senza dover leggere il codice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Percorsi -------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent.parent          # .../code
RUNS_DIR = CODE_DIR / "runs"
EDA_SUMMARY_PATH = CODE_DIR / "eda_outputs" / "dermamnist_eda_summary.json"
NPZ_PATH = Path.home() / ".medmnist" / "dermamnist.npz"

# --- Costanti -------------------------------------------------------------------------
NUM_CLASSES = 7
CLASS_ABBR = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

# Statistiche del pre-addestramento ImageNet: sono le costanti attese da una ResNet-18
# pre-addestrata. Le costanti specifiche di DermaMNIST vivono in eda_outputs/ e si
# selezionano con normalization="dataset" (ha senso solo con pretrained=False).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUGMENTATIONS = ("none", "d4")
CLASS_WEIGHTINGS = ("none", "balanced")
NORMALIZATIONS = ("imagenet", "dataset")


@dataclass(frozen=True)
class TrainConfig:
    """Tutto cio' che definisce un esperimento riproducibile.

    I valori di default corrispondono a un fine tuning completo di ResNet-18 pre-addestrata
    a 224x224, cioe' il preprocessing per cui i pesi ImageNet sono stati addestrati.
    """

    epochs: int = 3
    lr: float = 3e-4
    batch_size: int = 64
    input_size: int = 224
    augmentation: str = "none"        # none | d4
    class_weighting: str = "none"     # none | balanced
    normalization: str = "imagenet"   # imagenet | dataset
    pretrained: bool = True
    freeze_backbone: bool = False     # True = linear probe (solo la testa si addestra)
    seed: int = 42
    max_train_batches: int | None = None   # tronca l'epoca: per smoke test rapidi
    # Salvare i pesi costa ~45 MB per run, ma senza di essi il run non e' utilizzabile per
    # classificare nuove immagini: e' il default sensato per uno scheletro pensato per l'uso.
    save_weights: bool = True
    notes: str = ""

    def validate(self) -> None:
        """Solleva ValueError con un messaggio azionabile se la config non e' valida.

        Viene chiamata prima di ogni run: un agente che passa un valore fuori dominio deve
        ricevere subito un errore leggibile, non fallire a meta' training.
        """
        if self.epochs < 1:
            raise ValueError(f"epochs={self.epochs}: deve essere >= 1.")
        if not 0 < self.lr < 1:
            raise ValueError(f"lr={self.lr}: atteso un valore in (0, 1), tipicamente 1e-4..1e-2.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size={self.batch_size}: deve essere >= 1.")
        if self.input_size < 28:
            raise ValueError(f"input_size={self.input_size}: le immagini sorgente sono 28x28.")
        if self.augmentation not in AUGMENTATIONS:
            raise ValueError(
                f"augmentation='{self.augmentation}' non valida. Ammesse: {', '.join(AUGMENTATIONS)}."
            )
        if self.class_weighting not in CLASS_WEIGHTINGS:
            raise ValueError(
                f"class_weighting='{self.class_weighting}' non valido. "
                f"Ammessi: {', '.join(CLASS_WEIGHTINGS)}."
            )
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"normalization='{self.normalization}' non valida. Ammesse: {', '.join(NORMALIZATIONS)}."
            )

    def to_dict(self) -> dict:
        return asdict(self)


def load_eda_summary() -> dict:
    """Carica il riepilogo prodotto dall'EDA.

    E' il contratto fra l'analisi esplorativa e il training: costanti di normalizzazione,
    conteggi per classe e class weight vengono da li', non ricalcolati.
    """
    if not EDA_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Riepilogo EDA non trovato in {EDA_SUMMARY_PATH}. "
            "Esegui l'ultima cella di dermaMNIST_eda.ipynb per generarlo."
        )
    with open(EDA_SUMMARY_PATH) as f:
        return json.load(f)
