"""I tool che l'agente puo' invocare.

Ogni tool e' un guscio sottile sopra una funzione deterministica del layer di training o
sopra `d4_augmentation.py`. Nessuna logica di dominio vive qui: se un tool sbaglia, il
problema e' riproducibile da riga di comando senza l'agente.

Due vincoli deliberati, dettati dall'uso di un modello locale:

1. **Argomenti piatti e tipizzati** (int, float, stringa da un insieme chiuso). I modelli
   piccoli sbagliano sistematicamente gli schemi annidati.
2. **Ritorno in testo breve e leggibile**, non JSON. Un modello da pochi miliardi di
   parametri estrae numeri da una frase molto piu' affidabilmente che da una struttura.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt

from d4_augmentation import D4_VARIANTS, contact_sheet, expand_d4, load_image, save_variants
from training.config import CODE_DIR, CLASS_ABBR, TrainConfig, load_eda_summary
from training.dataset import sample_image
from training.metrics import BASELINE_THRESHOLDS
from training.predict import CLASS_FULL_NAMES, classify_file, find_runs_with_weights
from training.predict import list_images as _list_images
from training.train import list_runs as _list_runs
from training.train import load_run, train_once

from .audit import audited

# Tetto di sicurezza: impedisce che una allucinazione del modello avvii un training di ore.
MAX_EPOCHS = 10

# Unica destinazione consentita per le immagini augmentate. L'agente non sceglie MAI dove
# scrivere: un modello da pochi miliardi di parametri che allucina un percorso non deve poter
# toccare il resto del filesystem.
AUGMENTED_DIR = CODE_DIR / "augmented"


def _safe_label(label: str) -> str:
    """Riduce un'etichetta ai soli caratteri alfanumerici, trattino e underscore.

    Serve sia per la sottocartella sia per il nome dei file: un `image_path` con '..' o
    separatori di percorso non deve poter far uscire la scrittura dalla directory dedicata,
    ne' produrre nomi di file bizzarri.
    """
    return "".join(ch for ch in label if ch.isalnum() or ch in "-_")[:40] or "immagine"


def _augmentation_outdir(label: str):
    """Crea e restituisce la sottocartella sicura sotto AUGMENTED_DIR."""
    outdir = AUGMENTED_DIR / _safe_label(label)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _augment_and_report(image, label: str, variant: str, source: str) -> str:
    """Applica la D4, scrive i file e compone la risposta testuale. Condivisa dai due tool."""
    variant = variant.strip().lower()
    if variant == "all":
        variants = expand_d4(image)
    elif variant in D4_VARIANTS:
        from d4_augmentation import apply_d4

        variants = {variant: apply_d4(image, variant)}
    else:
        return (
            f"Errore: variante '{variant}' non riconosciuta. "
            f"Usa 'all' oppure una fra: {', '.join(D4_VARIANTS)}."
        )

    outdir = _augmentation_outdir(label)
    written = save_variants(variants, outdir, stem=_safe_label(label))

    # Il provino ha senso solo per l'insieme completo: una griglia di una cella non serve.
    sheet_line = ""
    if len(variants) > 1:
        sheet_path = outdir / "provino.png"
        contact_sheet(variants, title=source).save(sheet_path)
        sheet_line = f"Provino d'insieme (apri questo per vederle tutte insieme): {sheet_path}\n"

    files = "\n".join(f"  {name}: {path}" for name, path in written.items())
    return (
        f"Augmentation D4 applicata a {source}.\n"
        f"Varianti generate: {len(variants)} ({', '.join(written)}).\n"
        f"{sheet_line}"
        f"File scritti in {outdir}:\n{files}\n"
        f"Le varianti sono permutazioni esatte dei pixel: nessuna perdita di informazione."
    )


def _resolve_run_id(run_id: str) -> str:
    """Traduce 'latest' nell'ultimo run. I modelli piccoli perdono facilmente gli ID."""
    if run_id.strip().lower() in ("latest", "ultimo", "last"):
        runs = _list_runs()
        if not runs:
            raise FileNotFoundError("Nessun run presente: eseguine uno con train_model.")
        return runs[0]["run_id"]
    return run_id.strip()


@tool
@audited
def inspect_dataset() -> str:
    """Riporta numerosita', distribuzione delle classi e sbilanciamento di DermaMNIST.

    Usalo prima di decidere la configurazione di un training.
    """
    s = load_eda_summary()
    counts = s["class_counts"]["train"]
    sizes = s["split_sizes"]
    per_class = ", ".join(
        f"{CLASS_ABBR[i]}={counts[i]}" for i in range(len(CLASS_ABBR))
    )
    weights = ", ".join(
        f"{CLASS_ABBR[i]}={s['class_weights_balanced'][i]:.2f}" for i in range(len(CLASS_ABBR))
    )
    return (
        f"DermaMNIST: {sum(sizes.values())} immagini 28x28 RGB, 7 classi di lesioni cutanee.\n"
        f"Split ufficiali: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}.\n"
        f"Immagini per classe nel train: {per_class}.\n"
        f"Sbilanciamento severo: rapporto {s['imbalance_ratio_train']}:1 fra la classe piu' "
        f"frequente (nv, nei benigni) e la piu' rara (df, dermatofibroma).\n"
        f"Class weight bilanciati disponibili: {weights}.\n"
        f"Conseguenza: l'accuracy e' fuorviante (predire sempre nv da' 0.669). "
        f"Valuta con balanced accuracy, macro-F1, AUC e recall su mel (melanoma)."
    )


@tool
@audited
def describe_augmentation() -> str:
    """Descrive l'augmentation disponibile per il training e quando conviene usarla."""
    return (
        f"Augmentation disponibile: 'd4' (gruppo diedrale, {len(D4_VARIANTS)} varianti: "
        f"{', '.join(D4_VARIANTS)}).\n"
        "Sono rotazioni di 90 gradi e specchiature: permutazioni esatte dei pixel, senza "
        "interpolazione ne' perdita di informazione.\n"
        "Sono sicure su queste immagini perche' le lesioni dermatoscopiche non hanno un "
        "orientamento canonico e sono centrate nel frame: ruotare non cambia la diagnosi.\n"
        "Per usarla in addestramento: train_model(augmentation='d4'); "
        "l'alternativa e' augmentation='none'.\n"
        "Per vederla applicata a un'immagine: augment_image su un file locale, oppure "
        "augment_dataset_sample su un campione del dataset. Entrambi salvano le varianti "
        "e un provino d'insieme."
    )


@tool
@audited
def train_model(
    epochs: int = 3,
    learning_rate: float = 0.0003,
    augmentation: str = "none",
    class_weighting: str = "none",
) -> str:
    """Addestra una ResNet-18 pre-addestrata su DermaMNIST e restituisce l'ID del run.

    Args:
        epochs: numero di epoche, da 1 a 10. Ogni epoca richiede circa 1 minuto.
        learning_rate: learning rate di Adam, tipicamente fra 0.0001 e 0.001.
        augmentation: 'none' oppure 'd4'.
        class_weighting: 'none' oppure 'balanced' per pesare le classi rare nella loss.
    """
    if not 1 <= epochs <= MAX_EPOCHS:
        return (
            f"Errore: epochs={epochs} fuori dall'intervallo consentito (1-{MAX_EPOCHS}). "
            f"Riprova con un valore in quell'intervallo."
        )
    cfg = TrainConfig(
        epochs=epochs,
        lr=learning_rate,
        augmentation=augmentation,
        class_weighting=class_weighting,
        notes="avviato dall'agente",
    )
    try:
        cfg.validate()
    except ValueError as exc:
        return f"Errore di configurazione: {exc}"

    summary = train_once(cfg, progress=lambda line: print(f"    {line}"))
    test = summary["test"]
    return (
        f"Training completato. run_id={summary['run_id']} "
        f"(epoche={epochs}, augmentation={augmentation}, class_weighting={class_weighting}, "
        f"durata={summary['total_seconds']}s).\n"
        f"Metriche sul test: balanced accuracy={test['balanced_accuracy']:.4f}, "
        f"macro-F1={test['macro_f1']:.4f}, AUC={test['auc_ovr']:.4f}, "
        f"recall melanoma={test['recall_mel']:.4f}, accuracy={test['accuracy']:.4f}.\n"
        f"Usa get_run_report('{summary['run_id']}') per il dettaglio per classe."
    )


@tool
@audited
def get_run_report(run_id: str = "latest") -> str:
    """Riporta le metriche di un training e le confronta con il baseline di riferimento.

    Args:
        run_id: l'ID restituito da train_model, oppure 'latest' per l'ultimo run.
    """
    try:
        resolved = _resolve_run_id(run_id)
        data = load_run(resolved)
    except FileNotFoundError as exc:
        return f"Errore: {exc}"

    test = data["test"]
    per_class = ", ".join(f"{k}={v:.3f}" for k, v in test["recall_per_class"].items())
    comparison = data["baseline_comparison"]
    verdict = "\n".join(
        f"  {name}: {c['valore']} contro soglia {c['soglia']} "
        f"({'supera' if c['supera'] else 'NON supera'}, delta {c['delta']:+})"
        for name, c in comparison.items()
    )
    return (
        f"Run {data['run_id']} | config: {data['config']['epochs']} epoche, "
        f"augmentation={data['config']['augmentation']}, "
        f"class_weighting={data['config']['class_weighting']}, lr={data['config']['lr']}.\n"
        f"Migliore epoca: {data['best_epoch']} (val balanced accuracy "
        f"{data['best_val_balanced_accuracy']:.4f}).\n"
        f"Test: balanced accuracy={test['balanced_accuracy']:.4f}, "
        f"macro-F1={test['macro_f1']:.4f}, AUC={test['auc_ovr']:.4f}, "
        f"accuracy={test['accuracy']:.4f}.\n"
        f"Recall per classe: {per_class}.\n"
        f"Confronto col baseline dell'analisi esplorativa:\n{verdict}"
    )


@tool
@audited
def list_runs() -> str:
    """Elenca i training gia' eseguiti, dal piu' recente, con le metriche principali."""
    runs = _list_runs()
    if not runs:
        return "Nessun training eseguito finora. Usa train_model per avviarne uno."
    lines = [
        f"  {r['run_id']}: {r['epochs']} epoche, augmentation={r['augmentation']}, "
        f"class_weighting={r['class_weighting']}, "
        f"balanced accuracy={r['test_balanced_accuracy']}, AUC={r['test_auc_ovr']}"
        for r in runs[:10]
    ]
    return f"Run eseguiti ({len(runs)} in totale, mostro i piu' recenti):\n" + "\n".join(lines)


@tool
@audited
def augment_image(image_path: str, variant: str = "all") -> str:
    """Applica l'augmentation D4 a un'immagine locale e salva il risultato su disco.

    Genera le 8 varianti (rotazioni e specchiature) piu' un provino d'insieme: una singola
    immagine a griglia, etichettata, comoda da guardare.

    Args:
        image_path: percorso del file immagine (.png, .jpg, .npy).
        variant: 'all' per tutte e otto, oppure il nome di una singola variante
            (r0, r90, r180, r270, m_r0, m_r90, m_r180, m_r270).
    """
    from pathlib import Path

    try:
        image = load_image(image_path)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        return f"Errore: {exc}"

    stem = Path(image_path).stem
    return _augment_and_report(image, stem, variant, Path(image_path).name)


@tool
@audited
def augment_dataset_sample(class_name: str, index: int = 0, variant: str = "all") -> str:
    """Applica l'augmentation D4 a un'immagine del dataset, scelta per classe.

    Utile per mostrare che effetto ha l'augmentation su un tipo di lesione, senza dover
    avere un file gia' pronto su disco.

    Args:
        class_name: sigla della classe: akiec, bcc, bkl, df, mel, nv, vasc.
        index: quale immagine di quella classe usare, a partire da 0.
        variant: 'all' per tutte e otto, oppure il nome di una singola variante.
    """
    try:
        image = sample_image(class_name, index)
    except (ValueError, FileNotFoundError) as exc:
        return f"Errore: {exc}"

    name = class_name.strip().lower()
    return _augment_and_report(
        image, f"dataset_{name}_{index}", variant, f"campione {index} della classe {name}"
    )


@tool
@audited
def list_images_in_directory(directory: str) -> str:
    """Elenca le immagini presenti in una directory locale.

    Usalo quando l'utente indica una cartella invece di un singolo file, per sapere
    quali immagini puoi classificare.

    Args:
        directory: percorso della cartella, ad esempio './mie_immagini' o '~/Desktop/lesioni'.
    """
    try:
        paths = _list_images(directory)
    except (FileNotFoundError, ValueError) as exc:
        return f"Errore: {exc}"
    if not paths:
        return f"Nessuna immagine trovata in '{directory}'."
    listed = "\n".join(f"  {p}" for p in paths[:30])
    extra = f"\n(e altre {len(paths) - 30})" if len(paths) > 30 else ""
    return f"Trovate {len(paths)} immagini in '{directory}':\n{listed}{extra}"


@tool
@audited
def classify_image(image_path: str, run_id: str = "latest") -> str:
    """Classifica una singola immagine di lesione cutanea con un modello gia' addestrato.

    Restituisce la classe predetta e la probabilita' di tutte e 7 le classi.

    Args:
        image_path: percorso del file immagine (.png, .jpg, .npy).
        run_id: quale modello usare; 'latest' e' il piu' recente con i pesi salvati.
    """
    try:
        result = classify_file(image_path, run_id)
    except FileNotFoundError as exc:
        return f"Errore: {exc}"
    except ValueError as exc:
        return f"Errore: {exc}"

    # Il nome esteso viene dato per OGNI classe elencata, non solo per la predetta: un modello
    # piccolo, se deve glossare da solo le sigle secondarie, se le inventa.
    ordered = "\n".join(
        f"  {k} ({CLASS_FULL_NAMES[k]}): {v:.1%}" for k, v in list(result["probabilities"].items())[:4]
    )
    return (
        f"Immagine: {result['image']}\n"
        f"Classe predetta: {result['predicted_class']} "
        f"({result['predicted_class_full']}) con confidenza {result['confidence']:.1%}.\n"
        f"Probabilita' principali:\n{ordered}\n"
        f"Modello usato: run {result['run_id']}.\n"
        f"Attenzione: il modello e' addestrato su immagini 28x28, quindi la predizione e' "
        f"indicativa e non ha valore diagnostico."
    )


@tool
@audited
def list_trained_models() -> str:
    """Elenca i modelli addestrati utilizzabili per classificare (quelli con i pesi salvati)."""
    runs = find_runs_with_weights()
    if not runs:
        return (
            "Nessun modello utilizzabile per la classificazione: nessun run ha i pesi salvati. "
            "Addestrane uno con train_model."
        )
    return f"Modelli disponibili per la classificazione ({len(runs)}): {', '.join(runs[:10])}."


@tool
@audited
def ask_human(question: str) -> str:
    """Chiede una decisione all'operatore umano e ne attende la risposta.

    Usalo quando non puoi decidere da solo: la richiesta e' ambigua, un risultato e'
    inatteso, oppure stai per fare qualcosa di costoso senza istruzioni chiare.

    Args:
        question: la domanda da porre, formulata in modo che si possa rispondere in una frase.
    """
    answer = interrupt({"type": "question", "question": question})
    return f"Risposta dell'operatore: {answer}"


TOOLS = [
    inspect_dataset,
    describe_augmentation,
    augment_image,
    augment_dataset_sample,
    train_model,
    get_run_report,
    list_runs,
    list_trained_models,
    list_images_in_directory,
    classify_image,
    ask_human,
]

__all__ = ["TOOLS", "MAX_EPOCHS", "AUGMENTED_DIR", "BASELINE_THRESHOLDS"]
