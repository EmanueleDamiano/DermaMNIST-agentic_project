"""
The predictor's two memories.

Deterministic memory - `ExecutionLog`
    An append-only JSONL file, one line per image per execution. Recall is an
    exact lookup by the image's SHA-256: the same bytes always bring back the
    same past records, nothing is ranked or guessed. It answers "have I seen
    this exact image before, what did I say, with which models?" - so the agent
    can flag when a new run contradicts an old one (e.g. after a new checkpoint
    joined the ensemble).

Stochastic memory - `KnowledgeBase`
    Retrieval over `database/clinical_kb.json` (StatPearls / PubMed chunks on dermoscopy and
    on the seven DermaMNIST conditions). Recall is by lexical similarity
    (TF-IDF, uni+bigrams), so what comes back depends on how the query is
    phrased and is only probably relevant - the reasoning step must treat it
    as background, not as evidence about the image.

TF-IDF rather than dense embeddings: 120 chunks, a technical vocabulary where
exact terms ("blue-white veil", "arborizing vessels") carry the signal, and no
model download or API key. Swapping in embeddings means replacing `search()`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tester.models import CLASS_NAMES, PROJECT_ROOT

import paths

DEFAULT_LOG_PATH = paths.PREDICTIONS_LOG
DEFAULT_CORPUS_PATH = paths.CLINICAL_KB


# ---------------------------------------------------------------------------
# Deterministic memory
# ---------------------------------------------------------------------------

class ExecutionLog:
    def __init__(self, path: str | Path = DEFAULT_LOG_PATH):
        self.path = Path(path)

    def _records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:       # a line cut by a crash
                    continue
        return out

    def lookup(self, sha256: str, limit: int = 5) -> list[dict]:
        """Past records for exactly this image, newest first."""
        hits = [r for r in self._records() if r.get("sha256") == sha256]
        return hits[::-1][:limit]

    def stats(self) -> dict:
        """A few numbers about the whole log, for context."""
        records = self._records()
        if not records:
            return {"executions": 0, "images": 0}
        return {
            "executions": len({r.get("execution_id") for r in records}),
            "images": len(records),
            "distinct_images": len({r.get("sha256") for r in records}),
            "overridden_by_reasoning": sum(bool(r.get("overridden")) for r in records),
            "low_confidence": sum(r.get("confidence_level") == "low" for r in records),
            "last_timestamp": records[-1].get("timestamp"),
        }

    def append(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compact_past(record: dict) -> dict:
    """The part of a past record worth showing the reasoning step."""
    return {
        "timestamp": record.get("timestamp"),
        "final_class": record.get("final_class"),
        "vote_class": record.get("vote_class"),
        "confidence_level": record.get("confidence_level"),
        "models": [m.get("name") for m in record.get("models", [])],
    }


# ---------------------------------------------------------------------------
# Stochastic memory
# ---------------------------------------------------------------------------

# Query expansion per class: the DermaMNIST label alone ("benign keratosis-like
# lesions") rarely matches how the literature talks about the lesion.
CLASS_QUERIES = {
    0: "actinic keratosis intraepithelial carcinoma Bowen disease squamous cell carcinoma "
       "in situ strawberry pattern glomerular vessels scale",
    1: "basal cell carcinoma arborizing vessels leaf-like areas blue-gray ovoid nests "
       "spoke wheel ulceration",
    2: "seborrheic keratosis solar lentigo lichen planus-like keratosis benign keratosis "
       "milia-like cysts comedo-like openings fissures ridges",
    3: "dermatofibroma central white patch scar-like area peripheral delicate pigment network",
    4: "melanoma atypical pigment network blue-white veil irregular dots globules streaks "
       "regression structures asymmetry",
    5: "melanocytic nevus benign mole reticular globular homogeneous pattern symmetric "
       "dysplastic nevus",
    6: "vascular lesion hemangioma cherry angioma red lacunae angiokeratoma",
}


class KnowledgeBase:
    def __init__(self, path: str | Path = DEFAULT_CORPUS_PATH, max_chars: int = 700):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.path = Path(path)
        self.max_chars = max_chars
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks: list[dict] = []
        for src in data.get("sources", []):
            for i, text in enumerate(src.get("chunks", [])):
                self.chunks.append({
                    "chunk_id": f"{src['id']}#{i}",
                    "source_id": src["id"],
                    "title": src.get("title"),
                    "citation": src.get("citation"),
                    "url": src.get("url"),
                    "text": text,
                })
        if not self.chunks:
            raise ValueError(f"no chunks in {self.path}")
        # Title prepended: "Seborrheic Keratosis" in a title is strong signal
        # for every chunk of that source, even ones that never repeat the name.
        docs = [f"{c['title']}. {c['text']}" for c in self.chunks]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                           stop_words="english", min_df=1)
        self._matrix = self._vectorizer.fit_transform(docs)

    def search(self, query: str, k: int = 3, exclude: set[str] = frozenset()) -> list[dict]:
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1]
        out = []
        for idx in order:
            if len(out) == k or scores[idx] <= 0:
                break
            c = self.chunks[idx]
            if c["chunk_id"] in exclude:
                continue
            text = c["text"]
            if len(text) > self.max_chars:
                text = text[: self.max_chars].rsplit(" ", 1)[0] + " ..."
            out.append({**c, "text": text, "score": round(float(scores[idx]), 4)})
        return out

    def evidence_for(self, class_ids: list[int], k_per_class: int = 2,
                     extra_query: str = "") -> list[dict]:
        """Chunks describing each candidate class, plus one on telling them apart."""
        seen: set[str] = set()
        out: list[dict] = []
        for cid in class_ids:
            query = f"{CLASS_NAMES[cid]} {CLASS_QUERIES[cid]} dermoscopy {extra_query}"
            for hit in self.search(query, k=k_per_class, exclude=seen):
                seen.add(hit["chunk_id"])
                out.append({**hit, "about_class": CLASS_NAMES[cid]})
        if len(class_ids) >= 2:
            a, b = CLASS_NAMES[class_ids[0]], CLASS_NAMES[class_ids[1]]
            for hit in self.search(f"differential diagnosis distinguish {a} versus {b} "
                                   f"dermoscopic criteria", k=1, exclude=seen):
                seen.add(hit["chunk_id"])
                out.append({**hit, "about_class": f"{a} vs {b}"})
        return out
