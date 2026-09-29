"""
Orchestrator: route the request, then run the right agents.

    START -> route -+-> tester -> reviewer -> [human_review] -> finalize -> END
                    |        \\___ (tester failed) _______________/
                    +-> trainer ------------------------------------------> END
                    +-> inspector (questions about the models) -------------> END
                    +-> [clarify] -> tester | trainer | inspector | END
                    +-> answer (nothing to do, e.g. a prediction without images) -> END

route         deterministic when it can be: an explicit `mode`, or images in the
              request, mean "predict" exactly as before this node existed (the
              prediction path makes no extra LLM call). Otherwise an LLM reads
              the request as structured output - intent plus only the training
              constraints the user actually stated - and when the intent is
              unclear the graph ASKS (interrupt) instead of guessing. No keyword
              lists.
trainer       `train_agent`: the training campaign, with its own human
              approvals (plan, proposals outside the autonomy envelope,
              promotion), which surface here as interrupts of this graph.
inspector     `models_agent`: answers questions about the available models
              (characteristics, validation metrics, ROC curves, the validation
              set, the rules of the vote). Read-only, no interrupt.

tester        `predict_agent`: ensemble prediction, vote, memory, KB, reasoning.
reviewer      `review_agent`: reads the tester's full trace and decisions, checks
              them against the vote, the memory and the knowledge base, and
              writes what the user reads.
human_review  only with `ask_human=True` and only if the reviewer flagged
              something: the graph pauses with LangGraph `interrupt()`, the
              caller shows the issues to a person and resumes with their
              decision (`Command(resume={...})`). Needs a checkpointer.
finalize      appends the review (and the human decision, if any) to
              runs_predict/reviews_log.jsonl, keyed by the tester's
              execution_id - next to the tester's own predictions_log.jsonl.

The two agents are compiled graphs of their own and are called as such: the
orchestrator hands the tester only `image_paths`/`request`/`messages`, and the
reviewer only the tester's final state. Either can be replaced, or run alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from predict_agent.models import PROJECT_ROOT

DEFAULT_REVIEW_LOG = PROJECT_ROOT / "runs_predict" / "reviews_log.jsonl"


class OrchestratorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    image_paths: list[str]
    request: str
    mode: str                  # "predict" | "train" | "models" | "" (let route decide)
    constraints: dict          # training constraints (from the caller or extracted by route)
    route: dict                # how the request was routed, and why
    train_result: dict         # trainer's final state, minus messages
    models_result: dict        # inspector's final state, minus messages
    test_result: dict          # tester's final state, minus messages
    review_result: dict        # reviewer's final state, minus messages
    needs_human_review: bool
    human_decision: dict
    report: str                # what the user reads
    error: str


ROUTE_SYSTEM = """\
You route requests on a dermatology research platform with three capabilities:
- predict: classify dermatoscopic IMAGES with the local models (needs images);
- train: train or improve the classification models (a training campaign);
- models: answer questions about the models already available and their data -
  which models there are, how they were trained, their accuracy, AUC or ROC
  curves, why one is excluded, how the vote or the promotion works, what the
  validation set (DermaMNIST-C val) or the datasets contain.
Return the intent: "predict", "train", "models", or "unclear" when the request
does not say which (greetings, questions about something else, ambiguous wording).
A question about the models is "models", not "train": "train" only when the
user asks to train or improve something.
For "train", extract ONLY constraints the user explicitly stated - never invent
values: arch (fpvit | resnet18 | efficientnet_b0 | convnext_tiny), max_trials,
max_minutes, max_epochs_per_run, patience, target_score (0-1), select_on
(balanced_acc | macro_f1 | macro_auc), autonomy (supervised | guarded | autonomous).
Map wording to values: "ask me before every run / let me confirm each one" ->
autonomy supervised; "do it on your own / don't ask me" -> autonomous; "a couple
of runs/experiments" -> max_trials 2; "half an hour" -> max_minutes 30.
"""


TRAIN_KEYS = ("arch", "max_trials", "max_minutes", "max_epochs_per_run", "patience",
              "target_score", "select_on", "autonomy")


def make_router(llm):
    """text -> the router LLM's structured answer (intent, reason, training constraints)."""
    from typing import Literal, Optional as Opt
    from pydantic import BaseModel, Field

    class Route(BaseModel):
        intent: Literal["predict", "train", "models", "unclear"]
        reason: str = Field(description="one short sentence")
        arch: Opt[Literal["fpvit", "resnet18", "efficientnet_b0", "convnext_tiny"]] = None
        max_trials: Opt[int] = None
        max_minutes: Opt[float] = None
        max_epochs_per_run: Opt[int] = None
        patience: Opt[int] = None
        target_score: Opt[float] = None
        select_on: Opt[Literal["balanced_acc", "macro_f1", "macro_auc"]] = None
        autonomy: Opt[Literal["supervised", "guarded", "autonomous"]] = None

    structured = llm.with_structured_output(Route)

    def route(text: str) -> dict:
        return structured.invoke([("system", ROUTE_SYSTEM), ("human", text)]).model_dump()

    return route


def route_text(router, text: str) -> dict:
    """Route a text request without images: {"route": ..., "constraints"?: ...}.

    Shared by the graph's route node and by callers that must know the intent
    before running the graph (the platform decides from it which lock a job
    needs: a question about the models must not queue behind a training campaign).
    """
    if router is None or not text.strip():
        return {"route": {"intent": "unclear", "by": "no_router",
                          "reason": "no images and no LLM to interpret the request"}}
    try:
        r = router(text)
    except Exception as exc:
        return {"route": {"intent": "unclear", "by": "router_error", "reason": str(exc)}}
    extracted = {k: r[k] for k in TRAIN_KEYS if r.get(k) is not None}
    out = {"route": {"intent": r["intent"], "by": "llm", "reason": r["reason"], "extracted": extracted}}
    if r["intent"] == "train":
        out["constraints"] = extracted
    return out


def _last_human_text(messages) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "human" or (isinstance(m, tuple) and m[0] in ("human", "user")):
            content = m.content if hasattr(m, "content") else m[1]
            return content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def build_orchestrator_graph(tester, reviewer, *, explain_process: bool = False,
                             ask_human: bool = False,
                             review_log: str | Path = DEFAULT_REVIEW_LOG,
                             checkpointer=None, trainer=None, router_llm=None,
                             inspector=None, min_balanced_acc: float = 0.0):
    """`trainer`: a compiled train_agent graph (compiled WITHOUT its own
    checkpointer, so it inherits this graph's and its interrupts resume here).
    `router_llm`: chat model for intent routing when there are no images.
    `inspector`: a compiled models_agent graph, for questions about the models."""
    review_log = Path(review_log)
    router = make_router(router_llm) if router_llm is not None else None

    def route(state: OrchestratorState) -> dict:
        from predict_agent.graph import paths_from_text
        text = state.get("request") or _last_human_text(state.get("messages"))
        mode = state.get("mode") or ""
        if mode in ("predict", "train", "models"):
            return {"route": {"intent": mode, "by": "caller", "reason": "explicit mode"}}
        if (state.get("route") or {}).get("intent"):
            return {"route": state["route"]}         # already routed by the caller (route_text)
        if state.get("image_paths") or paths_from_text(text):
            return {"route": {"intent": "predict", "by": "images",
                              "reason": "the request contains images"}}
        out = route_text(router, text)
        if "constraints" in out:
            out["constraints"] = {**out["constraints"], **(state.get("constraints") or {})}   # the caller wins
        return out

    def clarify(state: OrchestratorState) -> dict:
        answer = interrupt({"kind": "clarify_intent",
                            "question": ("It is not clear what you want to do: a prediction on images, a training run"
                                         + (", or information about the available models?" if inspector is not None
                                            else "?")),
                            "reason": state["route"].get("reason"),
                            "options": [{"value": "predict", "label": "Prediction on images"},
                                        {"value": "train", "label": "Train the models"}]
                                       + ([{"value": "models", "label": "Show the available models"}]
                                          if inspector is not None else [])
                                       + [{"value": "cancel", "label": "Cancel"}]})
        choice = answer.get("decision") if isinstance(answer, dict) else str(answer)
        if choice == "predict" and isinstance(answer, dict) and answer.get("image_paths"):
            return {"route": {**state["route"], "intent": "predict", "by": "human"},
                    "image_paths": answer["image_paths"]}
        return {"route": {**state["route"], "intent": choice if choice in ("predict", "train", "models")
                          else "cancel", "by": "human"}}

    def answer(state: OrchestratorState) -> dict:
        r = state.get("route", {})
        if r.get("intent") == "predict":
            msg = "A prediction needs images: attach them (or give their paths) and try again."
        elif r.get("intent") == "train" and trainer is None:
            msg = "The training agent is not configured in this orchestrator."
        elif r.get("intent") == "models" and inspector is None:
            msg = "The models agent is not configured in this orchestrator."
        else:
            msg = "Request cancelled."
        return {"report": msg, "messages": [AIMessage(content=msg, name="derma_orchestrator")]}

    def run_trainer(state: OrchestratorState) -> dict:
        text = state.get("request") or _last_human_text(state.get("messages"))
        out = trainer.invoke({"request": text, "constraints": state.get("constraints") or {}})
        result = {k: v for k, v in out.items() if k != "messages"}
        return {"train_result": result, "report": out.get("report", ""),
                "messages": [AIMessage(content=out.get("report", ""), name="derma_trainer")]}

    def run_inspector(state: OrchestratorState) -> dict:
        text = state.get("request") or _last_human_text(state.get("messages"))
        out = inspector.invoke({"request": text, "min_balanced_acc": min_balanced_acc})
        result = {k: v for k, v in out.items() if k != "messages"}
        return {"models_result": result, "report": out.get("report", ""), "error": out.get("error", ""),
                "messages": [AIMessage(content=out.get("report", ""), name="derma_models")]}

    def after_route(state: OrchestratorState) -> str:
        intent = state["route"]["intent"]
        if intent == "predict":
            has_images = state.get("image_paths") or state["route"].get("by") in ("images", "caller")
            return "tester" if has_images else "answer"
        if intent == "train":
            return "trainer" if trainer is not None else "answer"
        if intent == "models":
            return "inspector" if inspector is not None else "answer"
        return "clarify"

    def after_clarify(state: OrchestratorState) -> str:
        intent = state["route"]["intent"]
        if intent == "train":
            return "trainer" if trainer is not None else "answer"
        if intent == "models":
            return "inspector" if inspector is not None else "answer"
        if intent == "predict" and state.get("image_paths"):
            return "tester"
        return "answer"

    def run_tester(state: OrchestratorState) -> dict:
        out = tester.invoke({"image_paths": state.get("image_paths") or [],
                             "request": state.get("request", ""),
                             "messages": state.get("messages", [])})
        result = {k: v for k, v in out.items() if k != "messages"}
        if out.get("error"):
            return {"test_result": result, "error": out["error"], "report": out["report"],
                    "messages": [AIMessage(content=out["report"], name="derma_predictor")]}
        return {"test_result": result, "error": ""}

    def run_reviewer(state: OrchestratorState) -> dict:
        test = state["test_result"]
        out = reviewer.invoke({"test": test, "request": test.get("request", ""),
                               "explain_process": explain_process})
        result = {k: v for k, v in out.items() if k != "messages"}
        return {"review_result": result, "needs_human_review": out["needs_human_review"],
                "report": out["report"],
                "messages": [AIMessage(content=out["report"], name="derma_reviewer")]}

    def human_review(state: OrchestratorState) -> dict:
        rv = state["review_result"]
        flagged = []
        for f, r, found in zip(state["test_result"]["final"], rv["review"]["images"], rv["checks"]):
            issues = [i for i in found + r.get("issues", [])
                      if i["severity"] in ("critical", "warning")]
            if issues:
                flagged.append({"file": f["file"], "final_class": f["final_class"],
                                "issues": issues})
        # Pauses the graph; the value passed to Command(resume=...) comes back here.
        decision = interrupt({"kind": "review_flags",
                              "question": "The reviewer flagged issues. "
                                          "Do you accept the predictions, reject them, or add a note?",
                              "flagged": flagged, "report": state["report"]})
        if not isinstance(decision, dict):
            decision = {"decision": str(decision)}
        decision = {**decision, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        note = f"\n\nHuman decision: {decision.get('decision')}" + (
            f" — {decision['note']}" if decision.get("note") else "")
        return {"human_decision": decision, "report": state["report"] + note}

    def finalize(state: OrchestratorState) -> dict:
        test, rv = state["test_result"], state["review_result"]
        record = {
            "execution_id": test.get("execution_id"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tester": test.get("reasoner"),
            "reviewer": rv.get("reviewer"),
            "needs_human_review": state.get("needs_human_review", False),
            "human_decision": state.get("human_decision"),
            "images": [{
                "file": f["file"], "sha256": f["sha256"], "final_class": f["final_class"],
                "confidence_level": f["confidence_level"],
                "verdict": r["verdict"],
                "automatic_checks": found,
                "reviewer_issues": r.get("issues", []),
                "decisive_model": {k: d[k] for k in ("model", "share_of_final_class_score",
                                                     "strongest_model", "strongest_model_agrees")},
            } for f, r, found, d in zip(test["final"], rv["review"]["images"], rv["checks"],
                                        rv["decisive"])],
            "process_summary": rv["review"].get("process_summary", ""),
            "overall": rv["review"].get("overall", ""),
            "reviewer_llm_error": rv.get("llm_error", ""),
        }
        review_log.parent.mkdir(parents=True, exist_ok=True)
        with open(review_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {}

    def after_tester(state: OrchestratorState) -> str:
        return "failed" if state.get("error") else "ok"

    def after_reviewer(state: OrchestratorState) -> str:
        return "ask" if ask_human and state.get("needs_human_review") else "done"

    g = StateGraph(OrchestratorState)
    g.add_node("route", route)
    g.add_node("clarify", clarify)
    g.add_node("answer", answer)
    if trainer is not None:
        g.add_node("trainer", run_trainer)
        g.add_edge("trainer", END)
    if inspector is not None:
        g.add_node("inspector", run_inspector)
        g.add_edge("inspector", END)
    g.add_node("tester", run_tester)
    g.add_node("reviewer", run_reviewer)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "route")
    extra = (["trainer"] if trainer is not None else []) + (["inspector"] if inspector is not None else [])
    g.add_conditional_edges("route", after_route, ["tester", "answer", "clarify"] + extra)
    g.add_conditional_edges("clarify", after_clarify, ["tester", "answer"] + extra)
    g.add_edge("answer", END)
    g.add_conditional_edges("tester", after_tester, {"ok": "reviewer", "failed": END})
    g.add_conditional_edges("reviewer", after_reviewer, {"ask": "human_review", "done": "finalize"})
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name="derma_orchestrator")


def build_orchestrator(model: Optional[str] = "claude-sonnet-5",
                       reviewer_model: Optional[str] = None,
                       *, verbose: int = 0, ask_human: bool = False,
                       explain_process: Optional[bool] = None,
                       predictor_kwargs: dict | None = None,
                       review_log: str | Path = DEFAULT_REVIEW_LOG,
                       checkpointer=None, with_trainer: bool = False,
                       trainer_model: Optional[str] = None):
    """Tester + reviewer, wired.

    Args:
        model: LLM of the testing agent ("provider:model"), None = vote only.
        reviewer_model: LLM of the reviewer; defaults to `model`. A different
            model makes the review more independent. None with model=None =
            automatic checks only.
        verbose: 0/1/2 as in predict_agent. With verbose >= 1 the reviewer
            also writes a plain-language account of what the pipeline did.
        explain_process: have the reviewer write the plain-language account
            of the run; defaults to `verbose > 0` (a UI can ask for it without
            the stderr trace).
        ask_human: pause for a human decision when the reviewer flags issues.
        with_trainer: add the training branch (train_agent), the models branch
            (models_agent, answering with `model`'s LLM) and intent routing.
            A checkpointer is then required (its approvals are interrupts); an
            InMemorySaver is created if none is given.
        trainer_model: LLM of the training agent (default: `model`); None with
            model=None means its deterministic policy.
            A checkpointer is required; an InMemorySaver is created if none given.
    """
    from predict_agent import build_predictor
    from predict_agent.agent import build_reasoning_llm
    from predict_agent.memory import DEFAULT_CORPUS_PATH, KnowledgeBase
    from review_agent import build_reviewer_graph

    reviewer_model = reviewer_model if reviewer_model is not None else model
    tester = build_predictor(model=model, verbose=verbose, **(predictor_kwargs or {}))
    corpus = (predictor_kwargs or {}).get("corpus_path", DEFAULT_CORPUS_PATH)
    reviewer = build_reviewer_graph(
        KnowledgeBase(corpus) if corpus else None,
        llm=build_reasoning_llm(reviewer_model) if reviewer_model else None,
        llm_name=reviewer_model or "", verbose=verbose,
    )
    trainer, router_llm, inspector = None, None, None
    if with_trainer:
        from models_agent import build_models_agent
        from train_agent.agent import build_trainer
        tmodel = trainer_model if trainer_model is not None else model
        trainer = build_trainer(tmodel or None, verbose=verbose)       # no checkpointer: inherits ours
        router_llm = build_reasoning_llm(tmodel) if tmodel else None
        inspector = build_models_agent(model or None)
    if (ask_human or with_trainer) and checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    if explain_process is None:
        explain_process = verbose > 0
    return build_orchestrator_graph(tester, reviewer, explain_process=explain_process,
                                    ask_human=ask_human, review_log=review_log,
                                    checkpointer=checkpointer, trainer=trainer, router_llm=router_llm,
                                    inspector=inspector,
                                    min_balanced_acc=(predictor_kwargs or {}).get("min_balanced_acc", 0.0))
