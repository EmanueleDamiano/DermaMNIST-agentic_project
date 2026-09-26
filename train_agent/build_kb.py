#!/usr/bin/env python3
"""
Build the training agent's knowledge base: train_agent/training_corpus.json.

Same schema as the clinical corpus.json (sources -> chunks), so the same
`predict_agent.memory.KnowledgeBase` indexes it. Two kinds of source, nothing
written by hand:

  paper    title, authors and abstract fetched from the arXiv API by id. Each
           id is checked against an expected title fragment; a mismatch (a
           wrong id, a withdrawn paper) is skipped and reported in `failures`,
           never silently included.
  project  this repository's own domain notes, extracted verbatim: the
           augmentation module's rationale and the per-field `note` of
           AUG_SEARCH_SPACE, and train.py's notes on imbalance and stopping.
           They are the project's decisions, cited as such.

    python -m train_agent.build_kb            # rebuild (network needed for arXiv)
"""

from __future__ import annotations

import ast
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).with_name("training_corpus.json")
ATOM = {"a": "http://www.w3.org/2005/Atom"}

# (arXiv id, expected title fragment, why it is in the KB)
PAPERS = [
    # --- the problem: DermaMNIST / HAM10000 -------------------------------
    ("2110.14795", "MedMNIST v2", "dataset, benchmark protocol and baselines"),
    ("1803.10417", "HAM10000", "source dataset, class distribution"),
    ("2401.14497", "Quality of DermaMNIST", "duplicates and leakage across splits"),
    ("1809.01442", "Data Augmentation for Skin Lesion Analysis", "which augmentations help skin lesion classifiers"),
    # --- class imbalance ----------------------------------------------------
    ("1901.05555", "Class-Balanced Loss", "effective-number class weighting"),
    ("1708.02002", "Focal Loss", "down-weighting easy majority examples"),
    ("1710.05381", "class imbalance problem in convolutional neural networks", "oversampling vs weighting in CNNs"),
    # --- optimisation and schedules ------------------------------------------
    ("1711.05101", "Decoupled Weight Decay", "AdamW"),
    ("1608.03983", "SGDR", "cosine schedule and warm restarts"),
    ("1506.01186", "Cyclical Learning Rates", "learning-rate range test"),
    ("1803.09820", "disciplined approach to neural network hyper-parameters", "lr, batch size, momentum, weight decay"),
    ("1706.02677", "Large Minibatch SGD", "linear lr scaling with batch size, warmup"),
    ("1609.04836", "Large-Batch Training", "generalisation gap of large batches"),
    ("1812.01187", "Bag of Tricks for Image Classification", "warmup, cosine decay, label smoothing"),
    # --- regularisation and augmentation --------------------------------------
    ("1708.04552", "Cutout", "occlusion regularisation"),
    ("1710.09412", "mixup", "interpolation regularisation"),
    ("1906.02629", "When Does Label Smoothing Help", "label smoothing and calibration"),
    # --- architectures the agent can choose ------------------------------------
    ("1512.03385", "Deep Residual Learning", "ResNet"),
    ("1905.11946", "EfficientNet", "EfficientNet"),
    ("2201.03545", "ConvNet for the 2020s", "ConvNeXt"),
    ("2010.11929", "An Image is Worth 16x16 Words", "ViT and its data hunger"),
    ("2106.10270", "How to train your ViT", "augmentation and regularisation for ViTs"),
]


def _ssl_context():
    """Verified TLS. The python.org macOS build ships without CA certificates
    ("Install Certificates.command"); certifi's bundle fills the gap when present."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_arxiv(ids: list[str]) -> dict[str, dict]:
    url = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids)
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as r:
        root = ET.fromstring(r.read())
    out = {}
    for e in root.findall("a:entry", ATOM):
        raw_id = e.find("a:id", ATOM).text.rsplit("/abs/", 1)[-1]
        base = re.sub(r"v\d+$", "", raw_id)
        title = " ".join((e.find("a:title", ATOM).text or "").split())
        out[base] = {
            "title": title,
            "summary": " ".join((e.find("a:summary", ATOM).text or "").split()),
            "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
            "year": (e.find("a:published", ATOM).text or "")[:4],
        }
    return out


def _cite(authors: list[str], year: str) -> str:
    first = authors[0].split()[-1] if authors else "?"
    return f"{first} et al., arXiv {year}" if len(authors) > 2 else \
        f"{' and '.join(a.split()[-1] for a in authors)}, arXiv {year}"


def paper_sources() -> tuple[list[dict], list[dict]]:
    sources, failures = [], []
    fetched: dict[str, dict] = {}
    ids = [p[0] for p in PAPERS]
    for attempt in range(3):                      # the API rate-limits bursts
        try:
            fetched = fetch_arxiv(ids)
            break
        except Exception as exc:
            failures.append({"stage": "fetch", "attempt": attempt + 1, "error": str(exc)})
            time.sleep(5 * (attempt + 1))
    for arxiv_id, fragment, why in PAPERS:
        meta = fetched.get(arxiv_id)
        if not meta:
            failures.append({"id": arxiv_id, "reason": "not returned by arXiv"})
            continue
        if fragment.lower() not in meta["title"].lower():
            failures.append({"id": arxiv_id, "reason": f"title mismatch: got {meta['title']!r}, "
                                                        f"expected to contain {fragment!r}"})
            continue
        sources.append({
            "id": f"arXiv:{arxiv_id}", "kind": "paper", "title": meta["title"],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "citation": _cite(meta["authors"], meta["year"]),
            "topic": why, "chunks": [meta["summary"]],
        })
    return sources, failures


def _paragraphs(text: str, min_chars: int = 120) -> list[str]:
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) >= min_chars and not set(p) <= set("-= ")]
    # merge short neighbours so a chunk carries a whole argument
    out: list[str] = []
    for p in paras:
        if out and len(out[-1]) < 500:
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def project_sources() -> list[dict]:
    aug_path = PROJECT_ROOT / "fpvit" / "augment.py"
    tree = ast.parse(aug_path.read_text(encoding="utf-8"))
    aug_doc = ast.get_docstring(tree) or ""

    from fpvit.augment import AUG_SEARCH_SPACE
    notes = [f"Augmentation field {name} (range {spec.get('low', spec.get('choices'))}"
             f"{' to ' + str(spec['high']) if 'high' in spec else ''}, default {spec.get('default')}): "
             f"{spec['note']}" for name, spec in AUG_SEARCH_SPACE.items() if spec.get("note")]

    train_doc = ast.get_docstring(ast.parse((PROJECT_ROOT / "train.py").read_text(encoding="utf-8"))) or ""
    return [
        {"id": "project:augment", "kind": "project", "title": "DermaMNIST augmentation policy (fpvit/augment.py)",
         "url": "fpvit/augment.py", "citation": "this project, fpvit/augment.py",
         "topic": "label-preserving augmentation for dermoscopy", "chunks": _paragraphs(aug_doc)},
        {"id": "project:aug_search_space", "kind": "project",
         "title": "Augmentation search space and the domain reason for each bound",
         "url": "fpvit/augment.py#AUG_SEARCH_SPACE", "citation": "this project, AUG_SEARCH_SPACE",
         "topic": "bounds of each augmentation field",
         "chunks": ["\n".join(notes[i:i + 4]) for i in range(0, len(notes), 4)]},
        {"id": "project:train", "kind": "project", "title": "Training protocol notes: imbalance, selection, stopping (train.py)",
         "url": "train.py", "citation": "this project, train.py",
         "topic": "class imbalance levers, selection metric, early stopping", "chunks": _paragraphs(train_doc)},
    ]


def main() -> int:
    papers, failures = paper_sources()
    sources = papers + project_sources()
    OUT.write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "description": "Training knowledge base: arXiv abstracts (verified by id and title) "
                       "and this project's own domain notes.",
        "sources": sources, "failures": failures,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    n = sum(len(s["chunks"]) for s in sources)
    print(f"{OUT}: {len(sources)} sources ({len(papers)} papers), {n} chunks")
    for f in failures:
        print("  skipped:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
