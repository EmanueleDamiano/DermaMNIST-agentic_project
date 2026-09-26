"""Metriche di valutazione per un dataset fortemente sbilanciato.

L'accuracy non compare fra le metriche primarie per una ragione misurata: sul test set di
DermaMNIST predire sempre la classe maggioritaria (`nv`) da 0,669 di accuracy senza imparare
nulla. Le metriche che discriminano sono balanced accuracy, macro-F1 e AUC-OvR, piu' la
recall su `mel` (melanoma), che e' la classe a maggior costo clinico di errore.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .config import CLASS_ABBR, NUM_CLASSES

MEL_INDEX = CLASS_ABBR.index("mel")

# Soglie stabilite dall'EDA con una regressione logistica sui pixel grezzi (§8 del notebook).
# Sono il pavimento: un modello che non le supera non sta guadagnando nulla dalla CNN.
BASELINE_THRESHOLDS = {
    "balanced_accuracy": 0.4756,
    "macro_f1": 0.3577,
    "auc_ovr": 0.8613,
    "accuracy_majority_class": 0.6688,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calcola il quadro completo delle metriche per uno split.

    Args:
        y_true: etichette reali (N,).
        y_pred: etichette predette (N,).
        y_proba: probabilita' per classe (N, 7).
    """
    present = np.unique(y_true)
    # roc_auc_score in modalita' ovr richiede che tutte le classi siano rappresentate.
    if len(present) == NUM_CLASSES:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    else:
        auc = float("nan")

    recalls = []
    for k in range(NUM_CLASSES):
        mask = y_true == k
        recalls.append(float((y_pred[mask] == k).mean()) if mask.any() else float("nan"))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_ovr": auc,
        "recall_mel": recalls[MEL_INDEX],
        "recall_per_class": {CLASS_ABBR[k]: recalls[k] for k in range(NUM_CLASSES)},
        "support_per_class": {
            CLASS_ABBR[k]: int((y_true == k).sum()) for k in range(NUM_CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))
        ).tolist(),
    }


def compare_with_baseline(metrics: dict) -> dict:
    """Confronta le metriche di un run con le soglie dell'EDA.

    Restituisce, per ogni metrica, il valore, la soglia, il delta e l'esito.
    """
    out = {}
    for key, threshold in BASELINE_THRESHOLDS.items():
        if key == "accuracy_majority_class":
            value = metrics.get("accuracy", float("nan"))
        else:
            value = metrics.get(key, float("nan"))
        beats = bool(value > threshold) if not np.isnan(value) else False
        out[key] = {
            "valore": round(float(value), 4) if not np.isnan(value) else None,
            "soglia": threshold,
            "delta": round(float(value - threshold), 4) if not np.isnan(value) else None,
            "supera": beats,
        }
    return out


def format_metrics(metrics: dict) -> str:
    """Rende le metriche in una riga leggibile, per i log e per l'agente."""
    return (
        f"acc={metrics['accuracy']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_F1={metrics['macro_f1']:.4f} "
        f"AUC={metrics['auc_ovr']:.4f} "
        f"recall_mel={metrics['recall_mel']:.4f}"
    )
