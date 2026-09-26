"""
Augmentation D4 (gruppo diedrale) per immagini di lesioni cutanee — Livello A.

PERCHE' D4
----------
Le immagini dermatoscopiche non hanno un orientamento canonico (non esiste un "alto"
o un "basso" clinicamente definito) e nel dataset DermaMNIST la lesione e' sistematicamente
centrata nel frame. Le 8 simmetrie del quadrato — 4 rotazioni per {identita', specchiatura} —
sono quindi trasformazioni **label-preserving per costruzione**: una lesione ruotata di 90 gradi
resta esattamente la stessa diagnosi.

Sono inoltre **permutazioni esatte dei pixel**: nessuna interpolazione, nessun pixel di
riempimento ai bordi, nessuna perdita di informazione. A 28x28, dove ogni pixel conta, questa
proprieta' e' decisiva: e' l'unica augmentation attivabile senza doverne prima misurare
sperimentalmente il beneficio.

Riferimenti nell'EDA (`dermaMNIST_eda.ipynb`): §5 (lesione centrata, isotropia delle immagini
medie per classe) e la discussione sulle augmentation sicure a bassa risoluzione.

USO COME LIBRERIA
-----------------
    from d4_augmentation import expand_d4, random_d4, apply_d4

    variants = expand_d4(img)              # dict: 8 varianti, nome -> array
    aug, name = random_d4(img, seed=42)    # una variante casuale, riproducibile
    rotated = apply_d4(img, "r90")         # una variante specifica

USO DA RIGA DI COMANDO (pensato per invocazione da parte di un agente)
----------------------------------------------------------------------
    python d4_augmentation.py --describe
    python d4_augmentation.py --input lesione.png --outdir out/ --json
    python d4_augmentation.py --input lesione.png --variant random --seed 7 --outdir out/
    python d4_augmentation.py --input lesione.png --outdir out/ --contact-sheet
    python d4_augmentation.py --self-test

Dipendenze: numpy (obbligatoria), Pillow (solo per leggere/scrivere file immagine).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

__all__ = [
    "D4_VARIANTS",
    "apply_d4",
    "expand_d4",
    "random_d4",
    "load_image",
    "save_variants",
    "contact_sheet",
    "describe",
    "self_test",
]

# Le 8 trasformazioni del gruppo diedrale D4.
# Convenzione: "m_" indica una specchiatura orizzontale applicata PRIMA della rotazione;
# "rK" indica una rotazione di K gradi in senso antiorario.
D4_VARIANTS: tuple[str, ...] = (
    "r0",      # identita'
    "r90",
    "r180",
    "r270",
    "m_r0",    # sola specchiatura orizzontale
    "m_r90",
    "m_r180",
    "m_r270",
)

_ROTATION_STEPS = {"r0": 0, "r90": 1, "r180": 2, "r270": 3}

# Estensioni riconosciute come file immagine; tutto il resto viene tentato come .npy.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------------------
# Validazione
# --------------------------------------------------------------------------------------

def _validate_image(image: np.ndarray) -> np.ndarray:
    """Verifica che l'input sia un'immagine singola in formato HWC o HW.

    Solleva ValueError con un messaggio azionabile: chi chiama (agente incluso) deve poter
    capire dal solo messaggio come correggere l'input.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Atteso numpy.ndarray, ricevuto {type(image).__name__}. "
            "Usa load_image() per leggere un file, oppure converti l'input con np.asarray()."
        )
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Shape {image.shape}: atteso layout HWC con 1, 3 o 4 canali nell'ultima "
                "dimensione. Se l'array e' in formato CHW (canali per primi), trasponilo con "
                "np.transpose(image, (1, 2, 0))."
            )
        return image
    raise ValueError(
        f"Shape {image.shape} con {image.ndim} dimensioni: attesa una singola immagine "
        "(HW oppure HWC). Per un batch, applica la funzione a un'immagine alla volta."
    )


def _validate_variant(variant: str) -> str:
    if variant not in D4_VARIANTS:
        raise ValueError(
            f"Variante '{variant}' non riconosciuta. Valori ammessi: {', '.join(D4_VARIANTS)}."
        )
    return variant


# --------------------------------------------------------------------------------------
# Trasformazioni
# --------------------------------------------------------------------------------------

def apply_d4(image: np.ndarray, variant: str) -> np.ndarray:
    """Applica una singola trasformazione del gruppo D4.

    Args:
        image: immagine come array HW (scala di grigi) o HWC (1, 3 o 4 canali).
            Qualsiasi dtype: i valori non vengono mai modificati, solo riposizionati.
        variant: uno dei nomi in `D4_VARIANTS`.

    Returns:
        Un nuovo array contiguo con gli stessi pixel dell'input, riordinati.
        Su immagini non quadrate le rotazioni di 90 e 270 gradi scambiano altezza e larghezza.

    Nota implementativa: il risultato e' reso contiguo con `np.ascontiguousarray` perche'
    `np.rot90` e `np.fliplr` restituiscono viste con stride negativi, che `torch.from_numpy`
    rifiuta. Restituire array contigui evita l'errore a valle nella pipeline di training.
    """
    image = _validate_image(image)
    variant = _validate_variant(variant)

    out = image
    if variant.startswith("m_"):
        out = np.fliplr(out)          # specchiatura orizzontale (asse larghezza)
        variant = variant[2:]
    steps = _ROTATION_STEPS[variant]
    if steps:
        out = np.rot90(out, k=steps, axes=(0, 1))   # antiorario nel piano HW
    return np.ascontiguousarray(out)


def expand_d4(image: np.ndarray) -> dict[str, np.ndarray]:
    """Genera tutte e 8 le varianti D4 di un'immagine.

    Returns:
        Dizionario ordinato come `D4_VARIANTS`: nome della variante -> array.

    Le 8 varianti sono distinte per un'immagine generica, ma possono coincidere se l'immagine
    ha simmetrie proprie (ad esempio un'immagine uniforme produce 8 array identici).
    """
    return {name: apply_d4(image, name) for name in D4_VARIANTS}


def random_d4(
    image: np.ndarray,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, str]:
    """Estrae una variante D4 a caso, in modo riproducibile.

    Args:
        image: immagine HW o HWC.
        seed: se fornito, il risultato e' deterministico. Ignorato se si passa `rng`.
        rng: generatore gia' inizializzato, da preferire in un ciclo di training per non
            reinizializzare lo stato a ogni chiamata.

    Returns:
        (immagine trasformata, nome della variante applicata).

    Il nome della variante viene restituito insieme all'immagine perche' un agente che
    orchestra la pipeline deve poter registrare *quale* trasformazione e' stata applicata:
    senza questa informazione l'esperimento non e' tracciabile ne' riproducibile
    (requisito non funzionale di riproducibilita', WP1).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    variant = D4_VARIANTS[int(rng.integers(len(D4_VARIANTS)))]
    return apply_d4(image, variant), variant


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Carica un'immagine da file in un array HWC (o HW per la scala di grigi).

    Supporta i formati immagine comuni tramite Pillow e i file `.npy` tramite numpy.
    Le immagini con palette o con modalita' insolite vengono convertite in RGB.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    if path.suffix.lower() == ".npy":
        return _validate_image(np.load(path))

    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError(
            f"Estensione '{path.suffix}' non supportata. "
            f"Formati immagine: {', '.join(sorted(_IMAGE_SUFFIXES))}; oppure un file .npy."
        )

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow e' nel venv del progetto
        raise ImportError(
            "Pillow serve per leggere file immagine. Installalo con `pip install pillow`, "
            "oppure fornisci l'immagine come file .npy."
        ) from exc

    with Image.open(path) as im:
        if im.mode not in ("L", "RGB", "RGBA"):
            im = im.convert("RGB")
        return _validate_image(np.asarray(im))


def save_variants(
    variants: dict[str, np.ndarray],
    outdir: str | Path,
    stem: str = "image",
) -> dict[str, str]:
    """Salva le varianti su disco, una per file.

    Gli array `uint8` vengono scritti come PNG (senza perdita); qualsiasi altro dtype viene
    scritto come `.npy` per non alterare i valori con una conversione implicita.

    Returns:
        Dizionario nome della variante -> percorso del file scritto.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for name, arr in variants.items():
        if arr.dtype == np.uint8:
            from PIL import Image

            dest = outdir / f"{stem}__{name}.png"
            data = arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr
            Image.fromarray(data).save(dest)
        else:
            dest = outdir / f"{stem}__{name}.npy"
            np.save(dest, arr)
        written[name] = str(dest)
    return written


def _render_tile(array, tile_size: int):
    """Porta una variante alla dimensione del riquadro, scegliendo il ricampionamento giusto.

    Ingrandimento con NEAREST a fattore intero: la D4 e' una permutazione esatta dei pixel, e
    un ingrandimento sfocato suggerirebbe che l'augmentation interpoli, cioe' esattamente cio'
    che non fa. Un pixel dell'originale deve restare un blocco uniforme.

    Rimpicciolimento (immagini grandi) con LANCZOS: li' NEAREST produrrebbe aliasing e
    renderebbe il provino illeggibile.
    """
    from PIL import Image

    data = array[:, :, 0] if array.ndim == 3 and array.shape[2] == 1 else array
    img = Image.fromarray(data)
    longest = max(img.size)

    if longest <= tile_size:
        factor = max(1, tile_size // longest)
        return img.resize((img.width * factor, img.height * factor), Image.Resampling.NEAREST)

    scale = tile_size / longest
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def contact_sheet(
    variants: dict[str, np.ndarray],
    title: str | None = None,
    tile_size: int = 112,
    columns: int = 4,
):
    """Compone le varianti in un'unica immagine a griglia, etichettata.

    E' il formato che un umano apre davvero per capire a colpo d'occhio cosa ha prodotto
    l'augmentation: otto file separati costringono ad aprirli uno per uno.

    Args:
        variants: dizionario nome -> array, come restituito da `expand_d4`.
        title: intestazione facoltativa (tipicamente il nome dell'immagine di partenza).
        tile_size: lato del riquadro in pixel.
        columns: riquadri per riga.

    Returns:
        Un'immagine PIL pronta per `.save()`.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not variants:
        raise ValueError("Nessuna variante da comporre.")

    padding, caption_h = 10, 22
    font = ImageFont.load_default(size=13)
    title_font = ImageFont.load_default(size=15)

    names = list(variants)
    columns = max(1, min(columns, len(names)))
    rows = -(-len(names) // columns)          # divisione intera arrotondata per eccesso

    cell_w = tile_size + 2 * padding
    cell_h = tile_size + caption_h + 2 * padding
    title_h = 30 if title else 0

    sheet = Image.new("RGB", (columns * cell_w, title_h + rows * cell_h), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.text((padding, 8), title, fill=(30, 30, 30), font=title_font)

    for index, name in enumerate(names):
        row, col = divmod(index, columns)
        tile = _render_tile(variants[name], tile_size)

        # Riquadro centrato: le varianti r90/r270 di un'immagine non quadrata hanno lati scambiati.
        box_x = col * cell_w + padding
        box_y = title_h + row * cell_h + padding
        sheet.paste(tile, (box_x + (tile_size - tile.width) // 2,
                           box_y + (tile_size - tile.height) // 2))

        label = f"{name} (originale)" if name == "r0" else name
        text_w = draw.textbbox((0, 0), label, font=font)[2]
        draw.text(
            (col * cell_w + (cell_w - text_w) // 2, box_y + tile_size + 4),
            label,
            fill=(60, 60, 60),
            font=font,
        )

    return sheet


# --------------------------------------------------------------------------------------
# Introspezione e verifica — pensate per l'uso da parte di un agente
# --------------------------------------------------------------------------------------

def describe() -> dict:
    """Descrive le capacita' del modulo in forma strutturata.

    Serve a un agente che deve decidere se e come usare questo strumento senza leggere
    il codice sorgente.
    """
    return {
        "tool": "d4_augmentation",
        "purpose": (
            "Augmentation geometrica esatta (gruppo diedrale D4) per immagini dermatoscopiche: "
            "8 varianti label-preserving per immagine, senza interpolazione ne' perdita di pixel."
        ),
        "variants": list(D4_VARIANTS),
        "n_variants": len(D4_VARIANTS),
        "label_preserving": True,
        "lossless": True,
        "input": "array HW o HWC (1/3/4 canali), qualsiasi dtype; oppure file .png/.jpg/.npy",
        "output": "array con gli stessi pixel riordinati; su immagini non quadrate r90/r270 scambiano H e W",
        "functions": {
            "apply_d4(image, variant)": "applica una trasformazione specifica",
            "expand_d4(image)": "genera tutte e 8 le varianti",
            "random_d4(image, seed=None)": "estrae una variante casuale, restituisce anche il nome",
            "load_image(path)": "carica un'immagine da file",
            "save_variants(variants, outdir, stem)": "scrive le varianti su disco",
        },
        "notes": [
            "Adatta a lesioni centrate e prive di orientamento canonico (caso DermaMNIST).",
            "Non applicare a immagini in cui l'orientamento e' semanticamente rilevante.",
            "random_d4 restituisce il nome della variante: registrarlo per la riproducibilita'.",
        ],
    }


def self_test() -> dict:
    """Verifica le proprieta' che rendono D4 sicura. Restituisce l'esito dei singoli controlli.

    Un agente puo' invocarla per accertarsi che lo strumento funzioni prima di usarlo in
    una pipeline, invece di assumerlo.
    """
    checks: dict[str, bool] = {}
    rng = np.random.default_rng(0)

    # Immagine asimmetrica: garantisce che le 8 varianti siano effettivamente distinte.
    img = rng.integers(0, 256, size=(28, 28, 3), dtype=np.uint8)
    variants = expand_d4(img)

    checks["numero_varianti_corretto"] = len(variants) == 8
    checks["identita_invariata"] = np.array_equal(variants["r0"], img)
    checks["shape_preservata_su_immagine_quadrata"] = all(
        v.shape == img.shape for v in variants.values()
    )
    # Proprieta' centrale: ogni variante e' una permutazione dei pixel originali.
    reference = np.sort(img.ravel())
    checks["pixel_preservati"] = all(
        np.array_equal(np.sort(v.ravel()), reference) for v in variants.values()
    )
    checks["varianti_distinte"] = (
        len({v.tobytes() for v in variants.values()}) == 8
    )
    checks["array_contigui"] = all(v.flags["C_CONTIGUOUS"] for v in variants.values())

    # Struttura di gruppo: quattro rotazioni riportano all'identita'.
    quattro_rotazioni = img
    for _ in range(4):
        quattro_rotazioni = apply_d4(quattro_rotazioni, "r90")
    checks["r90_di_ordine_4"] = np.array_equal(quattro_rotazioni, img)
    checks["specchiatura_involutiva"] = np.array_equal(
        apply_d4(apply_d4(img, "m_r0"), "m_r0"), img
    )

    # Riproducibilita' di random_d4.
    a, name_a = random_d4(img, seed=123)
    b, name_b = random_d4(img, seed=123)
    checks["random_riproducibile"] = name_a == name_b and np.array_equal(a, b)

    # Immagini non quadrate e in scala di grigi: comportamento documentato, non crash.
    rect = rng.integers(0, 256, size=(20, 32), dtype=np.uint8)
    checks["scala_di_grigi_supportata"] = apply_d4(rect, "r180").shape == (20, 32)
    checks["non_quadrata_scambia_assi"] = apply_d4(rect, "r90").shape == (32, 20)

    # Gli errori devono essere informativi, non generici.
    try:
        apply_d4(img, "r45")
        checks["variante_invalida_rifiutata"] = False
    except ValueError:
        checks["variante_invalida_rifiutata"] = True
    try:
        apply_d4(np.zeros((2, 28, 28, 3)), "r0")
        checks["input_4d_rifiutato"] = False
    except ValueError:
        checks["input_4d_rifiutato"] = True

    return {"passed": all(checks.values()), "checks": checks}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="d4_augmentation.py",
        description="Applica l'augmentation D4 (8 simmetrie del quadrato) a un'immagine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi:\n"
            "  python d4_augmentation.py --describe\n"
            "  python d4_augmentation.py --self-test\n"
            "  python d4_augmentation.py --input lesione.png --outdir out/\n"
            "  python d4_augmentation.py --input lesione.png --variant random --seed 7 --json\n"
            "  python d4_augmentation.py --input lesione.png --outdir out/ --contact-sheet\n"
        ),
    )
    parser.add_argument("--input", "-i", help="Percorso dell'immagine (.png/.jpg/.npy).")
    parser.add_argument(
        "--variant",
        "-v",
        default="all",
        help="Variante da applicare: uno dei nomi D4, 'all' (default) o 'random'.",
    )
    parser.add_argument("--outdir", "-o", help="Cartella dove scrivere le varianti. Se omessa, non scrive nulla.")
    parser.add_argument("--seed", type=int, help="Seed per --variant random (rende l'esito riproducibile).")
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Scrive anche un provino d'insieme: una griglia etichettata di tutte le varianti. "
             "Richiede --outdir e ha senso solo con --variant all.",
    )
    parser.add_argument("--json", action="store_true", help="Stampa il risultato in JSON.")
    parser.add_argument("--describe", action="store_true", help="Stampa in JSON le capacita' del modulo ed esce.")
    parser.add_argument("--self-test", action="store_true", help="Esegue i controlli interni ed esce.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.describe:
        print(json.dumps(describe(), indent=2, ensure_ascii=False))
        return 0

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["passed"] else 1

    if not args.input:
        parser.error("serve --input (oppure usa --describe / --self-test).")

    try:
        image = load_image(args.input)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1

    try:
        if args.variant == "all":
            variants = expand_d4(image)
        elif args.variant == "random":
            aug, name = random_d4(image, seed=args.seed)
            variants = {name: aug}
        else:
            variants = {args.variant: apply_d4(image, args.variant)}
    except ValueError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1

    stem = Path(args.input).stem
    written = save_variants(variants, args.outdir, stem=stem) if args.outdir else {}

    sheet_path = None
    if args.contact_sheet:
        if not args.outdir:
            print(json.dumps(
                {"status": "error", "message": "--contact-sheet richiede anche --outdir."},
                ensure_ascii=False,
            ))
            return 1
        sheet_path = str(Path(args.outdir) / f"{stem}__provino.png")
        contact_sheet(variants, title=Path(args.input).name).save(sheet_path)

    result = {
        "status": "ok",
        "input": str(args.input),
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "n_variants": len(variants),
        "variants": [
            {
                "name": name,
                "shape": list(arr.shape),
                "path": written.get(name),
            }
            for name, arr in variants.items()
        ],
        "written_to": str(args.outdir) if args.outdir else None,
        "contact_sheet": sheet_path,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Input : {args.input}  shape={tuple(image.shape)} dtype={image.dtype}")
        print(f"Varianti generate: {len(variants)}")
        for name, arr in variants.items():
            dest = written.get(name)
            print(f"  {name:<8} shape={tuple(arr.shape)}" + (f"  -> {dest}" if dest else ""))
        if sheet_path:
            print(f"\nProvino d'insieme: {sheet_path}")
        if not args.outdir:
            print("\nNessun file scritto (--outdir non specificata).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
