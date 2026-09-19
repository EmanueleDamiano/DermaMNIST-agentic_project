"""Local retrieval over a small, versioned dermoscopy corpus."""

from __future__ import annotations

import json
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "knowledge_base"
CORPUS_PATH = CORPUS_DIR / "corpus.json"
USER_AGENT = "AgenticDerma/3.0 research-education"

SOURCES = (
    {
        "id": "NBK606113",
        "kind": "bookshelf",
        "title": "Dermatoscopic Characteristics of Melanoma Versus Benign Lesions and Nonmelanoma Cancers",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK606113/",
        "fetch_url": "https://www.ncbi.nlm.nih.gov/books/NBK606113/?report=xml",
        "citation": "Valenzuela and Hohnadel, StatPearls, updated 2024",
        "fallback": "Dermoscopy uses magnification and polarized light to show structures below the skin surface. Dermoscopic findings correlate with histopathology, but they remain part of a clinical assessment rather than an independent diagnosis.",
    },
    {
        "id": "NBK537131",
        "kind": "bookshelf",
        "title": "Dermoscopy Overview and Extradiagnostic Applications",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK537131/",
        "fetch_url": "https://www.ncbi.nlm.nih.gov/books/NBK537131/?report=xml",
        "citation": "Sonthalia, Yumeen, and Kaliyadan, StatPearls, updated 2023",
        "fallback": "Dermoscopy is a non-invasive in-vivo technique used to assess suspicious skin lesions. Its scope extends beyond pigmented lesions, although interpretation depends on the clinical setting and the structures visible in the image.",
    },
    {
        "id": "PMID12734496",
        "kind": "pubmed",
        "pmid": "12734496",
        "title": "Dermoscopy of pigmented skin lesions: results of a consensus meeting via the Internet",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12734496/",
        "citation": "Argenziano et al., Journal of the American Academy of Dermatology, 2003",
        "fallback": "The consensus procedure separates melanocytic from nonmelanocytic lesions before applying pattern analysis or a diagnostic algorithm. Agreement was stronger for complete methods than for many individual criteria, which supports structured interpretation of dermoscopic features.",
    },
    {
        "id": "PMID26896294",
        "kind": "pubmed",
        "pmid": "26896294",
        "title": "Standardization of terminology in dermoscopy: third consensus conference",
        "url": "https://pubmed.ncbi.nlm.nih.gov/26896294/",
        "citation": "Kittler et al., Journal of the American Academy of Dermatology, 2016",
        "fallback": "The International Society of Dermoscopy consensus standardizes the descriptive terms used for dermoscopic structures. Consistent terminology matters because a model label and a visible structure describe different levels of evidence.",
    },
    {
        "id": "PMID30106392",
        "kind": "pubmed",
        "pmid": "30106392",
        "title": "The HAM10000 dataset",
        "url": "https://pubmed.ncbi.nlm.nih.gov/30106392/",
        "citation": "Tschandl, Rosendahl, and Kittler, Scientific Data, 2018",
        "fallback": "HAM10000 contains 10,015 dermoscopic images collected from different populations and acquisition methods. Ground truth combines pathology, follow-up, expert consensus, and confocal microscopy, so the dataset supports benchmarking but does not reproduce a complete clinical encounter.",
    },
    {
        "id": "PMID36658144",
        "kind": "pubmed",
        "pmid": "36658144",
        "title": "MedMNIST v2",
        "url": "https://pubmed.ncbi.nlm.nih.gov/36658144/",
        "citation": "Yang et al., Scientific Data, 2023",
        "fallback": "MedMNIST v2 standardizes biomedical image datasets at 28 by 28 pixels for lightweight classification research. DermaMNIST therefore evaluates a constrained benchmark task; its output should not be interpreted as a substitute for clinical examination.",
    },
    {
        "id": "NBK545285",
        "kind": "bookshelf",
        "title": "Seborrheic Keratosis",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK545285/",
        "fetch_url": "https://www.ncbi.nlm.nih.gov/books/NBK545285/?report=xml",
        "citation": "Greco and Bhutta, StatPearls, updated 2024",
        "fallback": "Seborrheic keratosis is a common benign epidermal tumor with a stuck-on appearance. Dermoscopy typically shows milia-like cysts, comedo-like openings, and a sharply demarcated border, which distinguish it from a melanocytic or malignant lesion.",
    },
    {
        "id": "NBK563207",
        "kind": "bookshelf",
        "title": "Cherry Hemangioma",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK563207/",
        "fetch_url": "https://www.ncbi.nlm.nih.gov/books/NBK563207/?report=xml",
        "citation": "Qadeer, Singal, and Patel, StatPearls, updated 2023",
        "fallback": "Cherry hemangioma is the most common acquired vascular proliferation of the skin, presenting as a dome-shaped, ruby-colored papule. Dermoscopy characteristically shows red-to-purple lacunae, which separate it from pigmented or keratinocytic lesions.",
    },
    {
        "id": "NBK560606",
        "kind": "bookshelf",
        "title": "Dysplastic Nevus",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK560606/",
        "fetch_url": "https://www.ncbi.nlm.nih.gov/books/NBK560606/?report=xml",
        "citation": "Wensley and Zito, StatPearls, updated 2023",
        "fallback": "A melanocytic nevus is evaluated dermoscopically for symmetry, a single dominant pattern, and even pigment network, while asymmetric structures or multiple colors raise concern and prompt closer follow-up or biopsy rather than reassurance from pattern alone.",
    },
)

STOP_WORDS = {
    "about", "after", "also", "and", "are", "been", "before", "being", "between", "can", "does", "for",
    "from", "have", "into", "its", "more", "most", "not", "only", "other", "our", "that", "the", "their",
    "these", "this", "through", "used", "using", "was", "were", "what", "when", "where", "which", "with",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in STOP_WORDS]


class _BookshelfParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.block: list[str] = []
        self.blocks: list[str] = []
        self.block_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "div" and values.get("itemprop") == "text":
            self.depth = 1
            return
        if self.depth:
            if tag == "div":
                self.depth += 1
            if tag in {"h2", "h3", "p", "li"}:
                self.block_tag = tag
                self.block = []

    def handle_data(self, data: str) -> None:
        if self.depth and self.block_tag:
            self.block.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.depth:
            return
        if tag == self.block_tag:
            value = re.sub(r"\s+", " ", " ".join(self.block)).strip()
            if value:
                self.blocks.append(value)
            self.block_tag = None
            self.block = []
        if tag == "div":
            self.depth -= 1


def fetch(url: str, timeout: int = 25) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def bookshelf_text(source: dict[str, str]) -> str:
    parser = _BookshelfParser()
    parser.feed(fetch(source["fetch_url"]))
    kept: list[str] = []
    for block in parser.blocks:
        if block.lower() in {"references", "review questions", "figure", "figures"}:
            break
        if len(block) >= 45:
            kept.append(block)
    if not kept:
        raise ValueError(f"No chapter text found for {source['id']}")
    return "\n\n".join(kept)


def pubmed_text(source: dict[str, str]) -> str:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={source['pmid']}&retmode=xml"
    root = ET.fromstring(fetch(url))
    abstract: list[str] = []
    for node in root.findall(".//AbstractText"):
        text = "".join(node.itertext()).strip()
        label = node.attrib.get("Label", "").title()
        if text:
            abstract.append(f"{label}: {text}" if label else text)
    if not abstract:
        raise ValueError(f"No abstract found for {source['id']}")
    return "\n\n".join(abstract)


def chunk_text(text: str, limit: int = 1100) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in text.split("\n\n")]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 2 <= limit:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= limit:
            current = paragraph
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > limit and current:
                    chunks.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk) >= 80]


def refresh_corpus() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for source in SOURCES:
        try:
            text = bookshelf_text(source) if source["kind"] == "bookshelf" else pubmed_text(source)
        except (OSError, ValueError, ET.ParseError) as exc:
            text = source["fallback"]
            failures.append(f"{source['id']}: {exc}")
        records.append({
            "id": source["id"], "title": source["title"], "url": source["url"],
            "citation": source["citation"], "chunks": chunk_text(text),
        })
    corpus = {"created_at": now_iso(), "sources": records, "failures": failures}
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    return corpus


def load_corpus() -> dict[str, Any]:
    if CORPUS_PATH.is_file():
        try:
            return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return refresh_corpus()


class KnowledgeBase:
    def __init__(self) -> None:
        self.corpus = load_corpus()
        self.entries: list[dict[str, Any]] = []
        for source in self.corpus.get("sources", []):
            for index, text in enumerate(source.get("chunks", [])):
                words = tokens(f"{source['title']} {text}")
                self.entries.append({**source, "chunk": index, "text": text, "tokens": words})
        self.document_frequency = Counter()
        for entry in self.entries:
            self.document_frequency.update(set(entry["tokens"]))
        self.average_length = sum(len(entry["tokens"]) for entry in self.entries) / max(1, len(self.entries))

    def status(self) -> dict[str, Any]:
        return {
            "ready": bool(self.entries), "sources": len(self.corpus.get("sources", [])),
            "passages": len(self.entries), "updated_at": self.corpus.get("created_at"),
        }

    def retrieve(self, query: str, limit: int = 4) -> dict[str, Any]:
        query_terms = set(tokens(query))
        if not query_terms or not self.entries:
            return {"context": "", "sources": [], "passages": []}
        count = len(self.entries)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for entry in self.entries:
            frequencies = Counter(entry["tokens"])
            length = max(1, len(entry["tokens"]))
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                frequency_docs = self.document_frequency.get(term, 0)
                inverse = math.log(1 + (count - frequency_docs + 0.5) / (frequency_docs + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(1, self.average_length))
                score += inverse * frequency * 2.5 / denominator
            if score:
                ranked.append((score, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:limit]
        sources: list[dict[str, str]] = []
        context: list[str] = []
        passages: list[dict[str, str]] = []
        labels: dict[str, str] = {}
        for _, entry in selected:
            if entry["id"] not in labels:
                label = f"S{len(labels) + 1}"
                labels[entry["id"]] = label
                sources.append({
                    "label": label, "title": entry["title"], "url": entry["url"],
                    "citation": entry["citation"],
                })
            context.append(f"[{labels[entry['id']]}] {entry['title']}\n{entry['text']}")
            passages.append({"label": labels[entry["id"]], "title": entry["title"], "text": entry["text"]})
        return {"context": "\n\n".join(context), "sources": sources, "passages": passages}


if __name__ == "__main__":
    corpus = refresh_corpus()
    print(json.dumps({"path": str(CORPUS_PATH), "sources": len(corpus["sources"]), "failures": corpus["failures"]}, indent=2))
