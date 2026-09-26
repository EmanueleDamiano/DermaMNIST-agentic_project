"""
The reviewer agent: reads a finished prediction run and tells the user what it
means, or what is wrong with it.

    gather -> review -> render -> END

gather   deterministic: its OWN retrieval from the knowledge base (for the
         final class, and with the tester's rationale as the query - so the
         claims are checked against chunks the tester did not choose), the
         automatic checks (`checks.py`), and the decisive model per image.
review   the LLM reads the full trace (the same lines `-v` prints), the
         tester's decisions, the checks and the independent chunks, and
         answers per image: consistent, or issues found - with evidence.
         Without an LLM the checks alone are the review.
render   the text the user reads. If the run was verbose, it opens with a
         plain-language account of what the pipeline did.

The reviewer never re-classifies: like the tester, it does not see the image.
Its job is consistency - numbers vs trace, claims vs literature, decision vs
vote, today vs memory - and saying so to a human when something does not hold.
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from predict_agent.graph import DISCLAIMER
from predict_agent.memory import KnowledgeBase
from predict_agent.models import CLASS_NAMES
from review_agent.checks import decisive_model, run_checks

AGENT_NAME = "derma_reviewer"
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


class ReviewState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # --- input -----------------------------------------------------------------
    test: dict               # the predictor's final state (minus messages)
    request: str
    explain_process: bool    # write the step-by-step account (the user asked for verbose)
    # --- filled by the nodes ---------------------------------------------------
    reviewer_kb: list[list[dict]]
    checks: list[list[dict]]
    decisive: list[dict]
    llm_input: dict
    llm_output: dict
    llm_error: str
    reviewer: str
    review: dict             # the structured review, one entry per image
    needs_human_review: bool
    report: str


REVIEW_SYSTEM = f"""\
You are the reviewer in a two-agent dermatology research pipeline. A testing
agent has classified dermatoscopic images with an ensemble of local models
(FPViT on DermaMNIST, 7 classes: {json.dumps(dict(enumerate(CLASS_NAMES)))}),
voted, retrieved literature and written a rationale per image. Neither you nor
the tester can see the images.

You receive:
- `trace`: every step of the tester's run, exactly as printed to the user
  (inputs, memory hits, per-model probabilities, the vote, retrieved chunks,
  the tester LLM's raw answer).
- per image: the tester's final decision and rationale, the vote summary, the
  decisive model (the one contributing most to the final class's score, with
  its validation metrics), `automatic_checks` computed in code, the chunks the
  tester cited, and `independent_kb`: chunks YOU retrieved for the final class
  and for the rationale's own wording.

Your job, per image:
1. Consistency. Flag, with concrete evidence (quote numbers, chunk ids):
   - numbers in the rationale that do not match the trace;
   - clinical statements that contradict the literature chunks, or that cite a
     chunk which does not say what is claimed;
   - statements about what the image shows (the tester cannot see it);
   - a final class or confidence the vote does not support;
   - a past prediction for the same image that disagrees and is not explained.
   The automatic checks are facts computed in code and are shown to the user
   anyway: never contradict them, let them count in your verdict, but list in
   `issues` ONLY what they do not already say.
   Do not invent problems: if nothing is wrong, verdict is 'consistent'.
2. `user_summary`: 2-4 plain sentences for a non-specialist: the final class,
   how confident, the tester's reason restated simply, and the next step.
3. `decisive_model_explanation`: 1-3 sentences on which model drove the
   prediction and why it weighs that much (its skill = balanced accuracy above
   chance, its probability and validation precision for this class, how it was
   trained), and whether the strongest model agreed.
4. Severity: 'critical' when the answer shown to the user may be wrong in a
   way the rationale hides (misquoted numbers, a claim contradicted by the
   literature, a visual claim, an unjustified override, a melanoma/nevus close
   call presented as settled); 'warning' when a human should look; 'info'
   otherwise.

`process_summary`: if `explain_process` is true, 5-10 plain sentences telling
the user what the pipeline did, in order (what was read, memory, which models,
how they voted, what was retrieved, how the decision was made). Otherwise "".

Never re-classify an image yourself. Write in the language of the request;
Italian if there is no request.
"""


def _make_llm_reviewer(llm):
    from pydantic import BaseModel, Field

    class Issue(BaseModel):
        severity: Literal["critical", "warning", "info"]
        category: Literal["kb_contradiction", "vote_inconsistency", "memory_conflict",
                          "unsupported_claim", "numbers_mismatch", "model_reliability", "other"]
        description: str
        evidence: str = Field(description="numbers from the trace or chunk ids / quotes")

    class ImageReview(BaseModel):
        file: str
        verdict: Literal["consistent", "issues_found"]
        issues: list[Issue] = Field(default_factory=list)
        user_summary: str
        decisive_model_explanation: str

    class Review(BaseModel):
        images: list[ImageReview]
        process_summary: str = ""
        overall: str = Field(description="1-2 sentences over the whole batch")

    structured = llm.with_structured_output(Review)

    def review(payload: dict) -> dict:
        out = structured.invoke([("system", REVIEW_SYSTEM),
                                 ("human", json.dumps(payload, ensure_ascii=False, indent=1))])
        return out.model_dump()

    return review


def _deterministic_review(test: dict, checks, decisive) -> dict:
    images = []
    for f, found, d in zip(test["final"], checks, decisive):
        images.append({
            "file": f["file"],
            "verdict": "issues_found" if any(c["severity"] != "info" for c in found) else "consistent",
            "issues": [],
            "user_summary": (f"Classe proposta: {f['final_class']} (confidenza {f['confidence_level']}). "
                             f"Motivazione del tester: {f['rationale']}"),
            "decisive_model_explanation": _decisive_text(d),
        })
    return {"images": images, "process_summary": "", "overall": ""}


def _decisive_text(d: dict) -> str:
    agree = ("concorda" if d["strongest_model_agrees"]
             else f"non concorda (indica {d['strongest_model_top_class']})")
    return (f"{d['model']} porta il {d['share_of_final_class_score']:.0%} del punteggio della classe "
            f"finale: p={d['probability_for_final_class']:.2f}, skill {d['skill']:.2f} "
            f"(bal.acc val {d['val_balanced_acc']:.2f}), precision di val su questa classe "
            f"{d['val_precision_for_final_class']:.2f}; addestrato {d['epoch']} epoche, "
            f"{d['optimizer']}, class_weight={d['class_weight']}. "
            f"Il modello più forte ({d['strongest_model']}) {agree}.")


def build_review_payload(state: dict) -> dict:
    """Everything the reviewing LLM is shown, built from the state `review` receives.

    Module-level so that a UI can rebuild the exact context from the node's
    input the moment the node starts.
    """
    test = state["test"]
    tested_raw = {d["file"]: d for d in (test.get("llm_output") or {}).get("decisions", [])}
    return {
        "request": state.get("request", ""),
        "explain_process": bool(state.get("explain_process")),
        "trace": "\n".join(test.get("trace", [])),
        "tester_reasoner": test.get("reasoner"),
        "tester_summary": (test.get("llm_output") or {}).get("summary", ""),
        "images": [{
            "file": f["file"],
            "final": {k: f[k] for k in ("final_class", "confidence_level", "overridden",
                                        "rationale", "suggested_next_step")},
            "vote": {k: v[k] for k in ("vote_class", "vote_probability", "runner_up_class",
                                       "margin", "confidence_level", "soft_vote", "hard_vote",
                                       "models_agreeing_with_vote")},
            "decisive_model": d,
            "automatic_checks": found,
            "past": f.get("past_predictions", []),
            "cited_by_tester": [{k: c[k] for k in ("chunk_id", "citation", "text")}
                                for c in ev
                                if c["chunk_id"] in tested_raw.get(f["file"], {}).get("knowledge_used", [])],
            "independent_kb": [{k: c[k] for k in ("chunk_id", "about_class", "citation", "text")}
                               for c in kbc],
        } for f, v, d, found, ev, kbc in zip(test["final"], test["votes"], state["decisive"],
                                             state["checks"], test["evidence"],
                                             state["reviewer_kb"])],
    }


def build_reviewer_graph(kb: Optional[KnowledgeBase], llm=None, llm_name: str = "",
                         verbose: int = 0, checkpointer=None):
    reviewer_llm = _make_llm_reviewer(llm) if llm is not None else None

    def say(text: str) -> None:
        if verbose:
            print(text, file=sys.stderr, flush=True)

    def gather(state: ReviewState) -> dict:
        test = state["test"]
        reviewer_kb = []
        for f in test["final"]:
            chunks = []
            if kb is not None:
                seen: set[str] = set()
                for c in kb.evidence_for([f["final_class_id"]], k_per_class=2):
                    seen.add(c["chunk_id"])
                    chunks.append(c)
                for c in kb.search(f["rationale"], k=3, exclude=seen):
                    chunks.append({**c, "about_class": "rationale"})
            reviewer_kb.append(chunks)
        checks = run_checks(test, reviewer_kb)
        decisive = [decisive_model(f, v, test["predictions"], i, test["models"])
                    for i, (f, v) in enumerate(zip(test["final"], test["votes"]))]

        say("\n=== [reviewer: gather] " + "=" * 42)
        for f, found, d, kbc in zip(test["final"], checks, decisive, reviewer_kb):
            say(f"  {f['file']}: decisive model {d['model']} "
                f"({d['share_of_final_class_score']:.0%} of the final class score)")
            say(f"    independent KB: {[(c['chunk_id'], c['about_class'], c['score']) for c in kbc]}")
            for c in found or [{"severity": "-", "category": "-", "description": "no automatic finding"}]:
                say(f"    [{c['severity']}] {c['category']}: {c['description']}")
        return {"reviewer_kb": reviewer_kb, "checks": checks, "decisive": decisive}

    def review(state: ReviewState) -> dict:
        test = state["test"]
        if reviewer_llm is None:
            say("\n=== [reviewer: review] no LLM: the automatic checks are the review")
            return {"review": _deterministic_review(test, state["checks"], state["decisive"]),
                    "reviewer": "deterministic", "llm_input": {}, "llm_output": {}, "llm_error": ""}

        payload = build_review_payload(state)
        say("\n=== [reviewer: review] " + "=" * 42)
        say(f"  LLM: {llm_name}; payload {len(json.dumps(payload, ensure_ascii=False)):,} chars")
        if verbose >= 2:
            say(json.dumps(payload, ensure_ascii=False, indent=1))
        try:
            raw = reviewer_llm(payload)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            say(f"  LLM ERROR, automatic checks only: {err}")
            return {"review": _deterministic_review(test, state["checks"], state["decisive"]),
                    "reviewer": "deterministic", "llm_input": payload, "llm_output": {},
                    "llm_error": err}

        for r in raw["images"]:
            say(f"  {r['file']}: {r['verdict']}")
            for i in r["issues"]:
                say(f"    [{i['severity']}] {i['category']}: {i['description']} | {i['evidence']}")
        # Align with the tester's order; an image the LLM skipped gets the checks-only review.
        by_file = {r["file"]: r for r in raw["images"]}
        fallback = _deterministic_review(test, state["checks"], state["decisive"])["images"]
        images = [by_file.get(fb["file"], fb) for fb in fallback]
        return {"review": {**raw, "images": images}, "reviewer": llm_name or "llm",
                "llm_input": payload, "llm_output": raw, "llm_error": ""}

    def render(state: ReviewState) -> dict:
        test, rv = state["test"], state["review"]
        lines, flagged = [], 0
        if state.get("explain_process") and rv.get("process_summary"):
            lines += ["Cosa è successo", rv["process_summary"], ""]

        for f, r, found in zip(test["final"], rv["images"], state["checks"]):
            issues = sorted(found + r.get("issues", []), key=lambda i: SEVERITY_ORDER[i["severity"]])
            serious = [i for i in issues if i["severity"] in ("critical", "warning")]
            flagged += bool(serious)
            lines.append(f"{f['file']} → {f['final_class']} (confidenza {f['confidence_level']})"
                         + ("  ⚠ CRITICITÀ" if serious else "  ✓ coerente"))
            lines.append(f"  {r['user_summary']}")
            lines.append(f"  Modello decisivo: {r['decisive_model_explanation']}")
            for i in issues:
                ev = f" — {i['evidence']}" if i.get("evidence") else ""
                lines.append(f"  [{i['severity']}] {i['description']}{ev}")
            if f.get("sources"):
                lines.append("  Fonti: " + "; ".join(s["citation"] for s in f["sources"]))
            lines.append("")

        needs_human = flagged > 0
        if rv.get("overall"):
            lines.append(rv["overall"])
        lines.append(f"Esito revisione ({state['reviewer']}): {len(rv['images'])} immagini, "
                     f"{flagged} con criticità"
                     + (" → RICHIEDE REVISIONE UMANA." if needs_human
                        else " → nessuna contraddizione rilevata."))
        if state.get("llm_error"):
            lines.append(f"(revisione LLM fallita, solo controlli automatici: {state['llm_error']})")
        lines += ["", DISCLAIMER]
        report = "\n".join(lines)
        return {"report": report, "needs_human_review": needs_human,
                "messages": [AIMessage(content=report, name=AGENT_NAME)]}

    g = StateGraph(ReviewState)
    g.add_node("gather", gather)
    g.add_node("review", review)
    g.add_node("render", render)
    g.add_edge(START, "gather")
    g.add_edge("gather", "review")
    g.add_edge("review", "render")
    g.add_edge("render", END)
    return g.compile(checkpointer=checkpointer, name=AGENT_NAME)
