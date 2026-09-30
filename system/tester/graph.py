"""
The prediction agent as a LangGraph state machine.

    load_inputs -> recall_memory -> run_models -> vote -> retrieve_knowledge
                -> reason -> write_log -> END
          \\____________________ (no images / no models) ____________/-> finish

Only `reason` involves an LLM, and only to do what numbers cannot: weigh the
vote against the per-model evidence (which models are blind to which class,
what was said about this exact image before) and explain the call using the
knowledge base. Everything that decides *what the models said* is
deterministic Python, so the same image and the same checkpoints always give
the same vote, with or without an LLM.

The LLM may override the vote, but only towards a class in the vote's
candidate set (classes with real probability mass, or that some model voted
for). A class nobody supported is rejected in code and the vote stands - the
override is recorded either way, so the log shows where reasoning and vote
parted.

Multi-agent use: the state has a `messages` channel with the standard
`add_messages` reducer, so the compiled graph drops into a supervisor graph
built on `MessagesState` as a node, or as a subgraph; it reads the image paths
from `image_paths` or, failing that, from the last human message, and answers
with one AIMessage named "derma_predictor". Structured results stay in
`final` for the next agent to read without parsing prose.
"""

from __future__ import annotations

import json
import operator
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

from tester.trace import NodeTracer
from tester.memory import ExecutionLog, KnowledgeBase, compact_past
from tester.models import (CLASS_NAMES, PROJECT_ROOT, ModelZoo,
                                  collect_images, sha256_of)
from tester.voting import vote_one

AGENT_NAME = "derma_predictor"

DISCLAIMER = ("Research prototype trained on low-resolution dermoscopic images (DermaMNIST, "
              "28x28 and 224x224): not a diagnosis, and not a substitute for a dermatologist's "
              "examination.")


def resize_info(size: list[int], cards: dict) -> dict:
    """How each input-size group of the ensemble sees an image of `size`, and
    which models therefore sit out the vote on it.

    Every model resizes the image to its own input (28 or 224 px).
    Downsampling is how the dataset itself was built from the originals, so it
    is only noted. Upsampling is not: measured on the DermaMNIST-C test, the
    224 px models shown 28 px images upsampled to 224 predict "nevus" for 95 %
    of them (balanced accuracy 0.21-0.27), and with their high vote weight
    they drag the whole ensemble down from 0.53 to 0.28. So a model that would
    have to upsample the image does not vote on it - unless every model would,
    in which case all vote and the image is flagged as out of distribution.
    """
    w, h = size
    by_size: dict[int, list[str]] = {}
    for card in cards.values():
        by_size.setdefault(card.input_size, []).append(card.name)
    up, down, notes = [], [], []
    for s, names in sorted(by_size.items()):
        if [w, h] == [s, s]:
            continue
        if max(w, h) < s:
            up += names
        else:
            down += names
            notes.append(f"downsampled from {w}x{h} to {s}x{s} for {', '.join(names)}, "
                         "as the dataset images were")
    all_up = bool(up) and len(up) == len(cards)
    excluded = [] if all_up else up
    if excluded:
        notes.append(f"{w}x{h} is below the {', '.join(sorted({f'{cards[n].input_size}x{cards[n].input_size}' for n in excluded}))} "
                     f"input of {', '.join(excluded)}: upsampled, it would show them less detail than "
                     "they were trained on, so they do not vote on this image")
    elif all_up:
        notes.append(f"{w}x{h} is below the input of every model: all of them see it upsampled, "
                     "out of distribution for the whole ensemble")
    return {"resized": bool(up or down), "upsampled_for": up, "downsampled_for": down,
            "excluded_from_vote": excluded, "all_upsampled": all_up,
            "resize_note": "; ".join(notes)}


class PredictorState(TypedDict, total=False):
    # --- shared with a supervisor ------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    # --- input ---------------------------------------------------------------
    image_paths: list[str]
    request: str                     # optional free text: what the caller wants
    # --- filled by the nodes -----------------------------------------------
    images: list[dict]               # file, path, sha256, original_size, resize_info()
    past: dict[str, list[dict]]      # sha256 -> past records (deterministic memory)
    log_stats: dict
    models: list[dict]               # public ModelCards of the ensemble
    predictions: dict[str, list[list[float]]]
    votes: list[dict]                # one per image, aligned with `images`
    evidence: list[list[dict]]       # KB chunks per image (stochastic memory)
    final: list[dict]                # one per image: the answer
    reasoner: str                    # which LLM decided, or "deterministic"
    execution_id: str                # key of this run's lines in the log
    llm_input: dict                  # exactly what the reasoning LLM was sent
    llm_output: dict                 # its raw structured answer, before validation
    llm_error: str                   # set if the LLM call failed (vote was kept)
    rejected_overrides: list[dict]   # LLM choices refused in code
    trace: Annotated[list[str], operator.add]   # what -v prints, always collected
    report: str
    error: str


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_PATH_RE = re.compile(
    r"""'([^']+)'|"([^"]+)"|(\S+\.(?:png|jpe?g|bmp|tiff?))(?=[\s,;)]|$)|(\S+/)""",
    re.IGNORECASE,
)


def paths_from_text(text: str) -> list[str]:
    """Image files or folders mentioned in free text. Quote paths with spaces."""
    found = []
    for m in _PATH_RE.finditer(text):
        raw = next(g for g in m.groups() if g)
        for base in (Path.cwd(), PROJECT_ROOT):
            p = Path(raw).expanduser()
            p = p if p.is_absolute() else base / p
            if p.exists():
                found.append(str(p))
                break
    return found


def _last_human_text(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            content = m.content
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# The reasoning step
# ---------------------------------------------------------------------------

REASON_SYSTEM = f"""\
You are the prediction agent of a dermatology research system. Local image
classifiers (FPViT models trained on DermaMNIST, 7 classes) have already looked
at each image and voted. You do NOT see the images. Your job is to turn the
vote into a final answer per image, and to explain it.

Classes: {json.dumps(dict(enumerate(CLASS_NAMES)))}

You receive, per image:
- `vote`: the skill-weighted soft vote (the default answer), the precision-
  weighted hard vote, the most confident single model, the margin between the
  first two classes, a starting confidence level, and every model's opinion
  with its validation skill and the classes it is blind to (val recall 0.0).
- `past`: exact past predictions for this same image (same SHA-256), if any.
- `knowledge`: literature chunks retrieved for the leading classes. They
  describe what those conditions look like; they are NOT observations of this
  image, and you must never claim a feature is present in it.

How to decide:
1. Start from the soft vote. Keep it unless the evidence says otherwise.
2. Discount dissent from a model that is blind to the voted class: it could
   not have voted for it. Discount confidence from a low-skill model.
3. The training data is 67% melanocytic nevi: a vote for nevi from weak models
   is the prior talking. Melanoma vs nevus is the costly confusion - if they
   are the top two and the margin is small, say so explicitly.
4. You may choose a different class ONLY from `candidate_class_ids`, and only
   with a concrete reason from the vote evidence or the past records. Set
   `overridden` true when you do.
5. If a past record for this image disagrees with the current vote, mention it
   (the ensemble has probably changed since).
6. Confidence: 'high' | 'medium' | 'low'. Never above the starting level unless
   you explain why; lower it freely.

Write rationales in the language of the request (English if unsure), 2-4
sentences, quoting the numbers you were given. Cite knowledge chunks by their
`chunk_id` in `knowledge_used` only if your rationale actually uses them.
"""


def build_reason_payload(state: dict) -> dict:
    """Everything the reasoning LLM is shown, built from the state `reason` receives.

    Module-level so that a UI can rebuild the exact context from the node's
    input the moment the node starts, without waiting for it to finish.
    """
    return {
        "request": state.get("request", ""),
        "ensemble": [{k: m[k] for k in ("name", "epoch", "val_balanced_acc",
                                        "skill", "blind_classes")}
                     for m in state["models"]],
        "images": [{
            "file": img["file"],
            "resized_from": img["original_size"] if img["resized"] else None,
            "resize_note": img.get("resize_note", ""),
            "vote": vote_,
            "past": state["past"].get(img["sha256"], []),
            "knowledge": [{k: c[k] for k in ("chunk_id", "about_class", "citation", "text")}
                          for c in ev],
        } for img, vote_, ev in zip(state["images"], state["votes"], state["evidence"])],
    }


def _make_llm_reasoner(llm):
    from pydantic import BaseModel, Field

    class ImageDecision(BaseModel):
        file: str = Field(description="the image file name, as given")
        # the range is checked in code (candidate set); numeric bounds are left out of
        # the schema because not every provider's structured output accepts them
        final_class_id: int = Field(description=f"class id, 0 to {len(CLASS_NAMES) - 1}")
        confidence_level: Literal["high", "medium", "low"]
        overridden: bool = Field(description="true if final_class_id differs from the vote")
        rationale: str
        knowledge_used: list[str] = Field(default_factory=list,
                                          description="chunk_id values actually used")
        suggested_next_step: str = Field(description="one short sentence")

    class Decisions(BaseModel):
        decisions: list[ImageDecision]
        summary: str = Field(description="2-3 sentences over the whole batch")

    from llm import structured as structured_output
    structured = structured_output(llm, Decisions)

    def reason(payload: dict) -> dict:
        out = structured.invoke([("system", REASON_SYSTEM),
                                 ("human", json.dumps(payload, ensure_ascii=False, indent=1))])
        return out.model_dump()

    return reason


def _deterministic_decision(img: dict, vote: dict) -> dict:
    notes = []
    if vote["confidence_level"] == "low":
        notes.append(f"close call, the gap to {vote['runner_up_class']} is only {vote['margin']:.2f}")
    if vote["models_blind_to_vote_class"]:
        notes.append("models that never predict this class on validation data: "
                     + ", ".join(vote["models_blind_to_vote_class"]))
    return {
        "file": img["file"],
        "final_class_id": vote["vote_class_id"],
        "confidence_level": vote["confidence_level"],
        "overridden": False,
        "rationale": (f"Weighted vote of the ensemble: {vote['vote_class']} with probability "
                      f"{vote['vote_probability']:.2f} (models agreeing: {vote['models_agreeing_with_vote']})."
                      + (" Note: " + "; ".join(notes) + "." if notes else "")),
        "knowledge_used": [],
        "suggested_next_step": ("dermatologist review" if vote["confidence_level"] != "high"
                                else "none from the model side"),
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_predictor_graph(zoo: ModelZoo,
                          log: ExecutionLog,
                          kb: Optional[KnowledgeBase],
                          llm=None,
                          llm_name: str = "",
                          checkpointer=None,
                          verbose: int = 0):
    """Compile the predictor. `llm=None` runs fully deterministic (no API key).

    `verbose`: 0 silent, 1 every node's inputs/outputs and the LLM's full answer
    on stderr, 2 also the complete JSON payload sent to the LLM.
    """
    reason_llm = _make_llm_reasoner(llm) if llm is not None else None
    tracer = NodeTracer(verbose, extra={"write_log": {"log_path": str(log.path)}})

    def load_inputs(state: PredictorState) -> dict:
        targets = list(state.get("image_paths") or [])
        request = state.get("request") or ""
        if not targets:
            text = _last_human_text(state.get("messages", []))
            request = request or text
            targets = paths_from_text(text)
        if not targets:
            return {"error": "no image paths given (pass `image_paths`, or put the "
                             "paths in the message, quoted if they contain spaces)",
                    "request": request}
        try:
            paths = collect_images(targets)
        except FileNotFoundError as exc:
            return {"error": str(exc), "request": request}
        if not paths:
            return {"error": f"no images found in {targets}", "request": request}
        images = []
        for p in paths:
            with Image.open(p) as im:
                size = list(im.size)
            images.append({"file": p.name, "path": str(p.resolve()),
                           "sha256": sha256_of(p), "original_size": size,
                           **resize_info(size, zoo.cards)})
        return {"images": images, "request": request, "error": ""}

    def recall_memory(state: PredictorState) -> dict:
        past = {img["sha256"]: [compact_past(r) for r in log.lookup(img["sha256"])]
                for img in state["images"]}
        return {"past": past, "log_stats": log.stats()}

    def run_models(state: PredictorState) -> dict:
        zoo.refresh()                              # pick up runs trained meanwhile
        if not zoo.cards:
            return {"error": f"no usable checkpoint under {[str(r) for r in zoo.roots]}"
                             + (f" (skipped: {zoo.skipped})" if zoo.skipped else "")}
        preds = zoo.predict([Path(img["path"]) for img in state["images"]])
        # Recomputed against the refreshed ensemble: a promotion since the
        # images were loaded may have added a model at another input size.
        images = [{**img, **resize_info(img["original_size"], zoo.cards)} for img in state["images"]]
        return {"predictions": preds, "images": images,
                "models": [c.public() for c in zoo.cards.values()]}

    def vote(state: PredictorState) -> dict:
        preds = state["predictions"]
        votes = [vote_one({m: preds[m][i] for m in preds
                           if m not in img.get("excluded_from_vote", [])}, zoo.cards)
                 for i, img in enumerate(state["images"])]
        return {"votes": votes}

    def retrieve_knowledge(state: PredictorState) -> dict:
        if kb is None:
            return {"evidence": [[] for _ in state["images"]]}
        cache: dict[tuple, list[dict]] = {}
        evidence = []
        for v in state["votes"]:
            key = (v["vote_class_id"], v["runner_up_class_id"])
            if key not in cache:
                cache[key] = kb.evidence_for(list(key))
            evidence.append(cache[key])
        return {"evidence": evidence}

    def reason(state: PredictorState) -> dict:
        images, votes = state["images"], state["votes"]
        decisions, summary, reasoner = None, "", "deterministic"
        reason_error = ""

        payload, raw = None, None
        if reason_llm is not None:
            payload = build_reason_payload(state)
            try:
                raw = reason_llm(payload)
                decisions, summary, reasoner = raw["decisions"], raw["summary"], llm_name or "llm"
            except Exception as exc:               # never lose the vote to an API error
                reason_error = f"{type(exc).__name__}: {exc}"

        by_file = {d["file"]: d for d in (decisions or [])}
        final, rejected = [], []
        for img, v, ev in zip(images, votes, state["evidence"]):
            d = by_file.get(img["file"]) or _deterministic_decision(img, v)
            chosen = d["final_class_id"]
            if chosen not in v["candidate_class_ids"]:
                rejected.append({"file": img["file"], "proposed_class_id": chosen,
                                 "candidates": v["candidate_class_ids"]})
                d = {**_deterministic_decision(img, v),
                     "rationale": (f"Reasoning proposed class {chosen}, which no model "
                                   f"supported; kept the vote. " + d["rationale"])}
                chosen = v["vote_class_id"]
            valid_chunks = {c["chunk_id"]: c for c in ev}
            final.append({
                "file": img["file"],
                "path": img["path"],
                "sha256": img["sha256"],
                "final_class_id": chosen,
                "final_class": CLASS_NAMES[chosen],
                "confidence_level": d["confidence_level"],
                "overridden": chosen != v["vote_class_id"],
                "vote_class": v["vote_class"],
                "vote_probability": v["vote_probability"],
                "runner_up_class": v["runner_up_class"],
                "margin": v["margin"],
                "most_confident_model": {k: v["most_confident_model"][k]
                                         for k in ("model", "predicted_class", "confidence")},
                "models_agreeing_with_vote": v["models_agreeing_with_vote"],
                "rationale": d["rationale"],
                "suggested_next_step": d["suggested_next_step"],
                "sources": [{"chunk_id": cid, "citation": valid_chunks[cid]["citation"],
                             "url": valid_chunks[cid]["url"]}
                            for cid in d.get("knowledge_used", []) if cid in valid_chunks],
                "resized_from": img["original_size"] if img["resized"] else None,
                "resize_note": img.get("resize_note", ""),
                "excluded_from_vote": img.get("excluded_from_vote", []),
                "all_upsampled": img.get("all_upsampled", False),
                "past_predictions": state["past"].get(img["sha256"], []),
            })

        report = _render_report(final, state["models"], summary, reasoner, reason_error)
        return {"final": final, "reasoner": reasoner, "report": report,
                "llm_input": payload or {}, "llm_output": raw or {},
                "llm_error": reason_error, "rejected_overrides": rejected,
                "messages": [AIMessage(content=report, name=AGENT_NAME)]}

    def write_log(state: PredictorState) -> dict:
        execution_id = uuid.uuid4().hex[:12]
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        models = [{"name": m["name"], "path": m["path"], "epoch": m["epoch"]}
                  for m in state["models"]]
        preds = state["predictions"]
        records = []
        for i, (f, v) in enumerate(zip(state["final"], state["votes"])):
            records.append({
                "execution_id": execution_id, "timestamp": ts,
                "request": state.get("request", ""), "reasoner": state["reasoner"],
                "file": f["file"], "path": f["path"], "sha256": f["sha256"],
                "models": models,
                "per_model_probabilities": {m: [round(p, 5) for p in preds[m][i]] for m in preds},
                "excluded_from_vote": f.get("excluded_from_vote", []),
                "vote_class": v["vote_class"], "vote_probability": v["vote_probability"],
                "margin": v["margin"],
                "final_class": f["final_class"], "confidence_level": f["confidence_level"],
                "overridden": f["overridden"], "rationale": f["rationale"],
            })
        log.append(records)
        return {"execution_id": execution_id}

    def finish_with_error(state: PredictorState) -> dict:
        msg = f"[{AGENT_NAME}] cannot predict: {state['error']}"
        return {"report": msg, "final": [], "messages": [AIMessage(content=msg, name=AGENT_NAME)]}

    def ok_or_error(state: PredictorState) -> str:
        return "error" if state.get("error") else "ok"

    g = StateGraph(PredictorState)
    for fn in (load_inputs, recall_memory, run_models, vote, retrieve_knowledge,
               reason, write_log, finish_with_error):
        g.add_node(fn.__name__, tracer.wrap(fn.__name__, fn))

    g.add_edge(START, "load_inputs")
    g.add_conditional_edges("load_inputs", ok_or_error,
                            {"ok": "recall_memory", "error": "finish_with_error"})
    g.add_edge("recall_memory", "run_models")
    g.add_conditional_edges("run_models", ok_or_error,
                            {"ok": "vote", "error": "finish_with_error"})
    g.add_edge("vote", "retrieve_knowledge")
    g.add_edge("retrieve_knowledge", "reason")
    g.add_edge("reason", "write_log")
    g.add_edge("write_log", END)
    g.add_edge("finish_with_error", END)
    return g.compile(checkpointer=checkpointer, name=AGENT_NAME)


def _render_report(final: list[dict], models: list[dict], summary: str,
                   reasoner: str, reason_error: str) -> str:
    lines = [f"Ensemble: {len(models)} local model(s) - "
             + ", ".join(f"{m['name']} (val bal.acc {m['val_balanced_acc']:.2f})" for m in models),
             f"Reasoning: {reasoner}"]
    if reason_error:
        lines.append(f"(LLM reasoning failed, fell back to the vote: {reason_error})")
    lines.append("")
    for f in final:
        head = (f"{f['file']}: {f['final_class']} - confidence {f['confidence_level']}"
                + (" [overrides vote]" if f["overridden"] else ""))
        lines.append(head)
        lines.append(f"  vote {f['vote_class']} p={f['vote_probability']:.2f}, runner-up "
                     f"{f['runner_up_class']} (margin {f['margin']:.2f}), agree "
                     f"{f['models_agreeing_with_vote']}; most confident: "
                     f"{f['most_confident_model']['model']} -> "
                     f"{f['most_confident_model']['predicted_class']} "
                     f"({f['most_confident_model']['confidence']:.2f})")
        lines.append(f"  {f['rationale']}")
        if f["past_predictions"]:
            prev = f["past_predictions"][0]
            lines.append(f"  previously ({prev['timestamp']}): {prev['final_class']}")
        if f["resized_from"]:
            kind = "warning" if f.get("all_upsampled") else "note"
            lines.append(f"  {kind}: image {f.get('resize_note') or 'resized'}")
        if f["sources"]:
            lines.append("  sources: " + "; ".join(s["citation"] for s in f["sources"]))
        lines.append("")
    if summary:
        lines += [summary, ""]
    lines.append(DISCLAIMER)
    return "\n".join(lines)
