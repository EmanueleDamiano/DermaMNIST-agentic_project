"""
Models agent: answers questions about the models available to the system.

    START -> collect_facts -> explain -> check_claims -> END

collect_facts  code (facts.py): characteristics, validation metrics, ROC curves
               and confusion matrices of the ensemble members and of the
               excluded runs, the composition of the validation set, the rules
               of the vote and of the promotion. Validation data only.
explain        the LLM answers the user's question from those facts alone.
               Without an LLM the answer is the deterministic summary.
check_claims   code: every number and every run-like name in the LLM's answer
               must appear in the facts (or in the question). If one does not,
               the answer shown is the deterministic summary, and the rejected
               claims are listed. The same rule the reviewer applies to the
               testing agent: numbers come from code, language from the LLM.

Read-only: no interrupt, no file written, no model touched.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from informer.facts import collect, summary

AGENT_NAME = "derma_models"

EXPLAIN_SYSTEM = """\
You answer questions about the classification models of a dermatology research
platform (DermaAgent, seven-class dermoscopic image classification on DermaMNIST).

You are given FACTS as JSON, computed in code: the ensemble members and the
excluded runs with their characteristics, their metrics on the common
validation set (DermaMNIST-C val), the composition of the datasets, the rules
of the vote and of the promotion gate, and caveats.

Rules:
- Answer ONLY from the facts. If they do not contain the answer, say so plainly
  and say what they do contain instead.
- Quote numbers exactly as they appear in the facts (same decimals). Do not
  compute new numbers: no differences, ratios, averages or rounded variants.
  A share such as 0.756 may be written as 75.6 %.
- Write model names exactly as in the facts.
- These are validation results: never present them as test results, and never
  invent test results.
- When a per-class number concerns a class with very few validation images
  (see validation_set.per_class), say how many images it rests on.
- When you discuss the ensemble's own metrics, mention they are optimistic.
- No medical advice.
- Answer in the language of the question, concisely: a few short paragraphs or
  a short list, plain text without markdown headings or tables. The platform
  shows the tables and ROC curves next to your answer.
"""

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%?)")
_IDENT = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")


class ModelsState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request: str
    min_balanced_acc: float
    facts: dict
    charts: dict
    answer: str                # what the user reads
    answered_by: str           # LLM name, or "deterministic"
    llm_output: str            # the LLM's raw answer, kept even when rejected
    llm_error: str
    claims: dict               # {"verdict": "supported" | "rejected" | "not_checked", "unsupported": [...]}
    report: str
    error: str


def build_explain_payload(state: dict) -> dict:
    """What the LLM is shown. Module-level so a UI can rebuild it exactly."""
    return {"question": state.get("request", ""), "facts": state.get("facts", {})}


def _numbers_in(obj) -> list[float]:
    out = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(n.replace(",", ".")) for n, _ in _NUMBER.findall(obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _numbers_in(k) + _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def check_claims(answer: str, facts: dict, question: str = "") -> list[str]:
    """The numbers and run-like names in `answer` that the facts do not contain.

    A number is supported if some fact equals it at the precision it is
    written with (0.76 matches 0.756), also read as a percentage (75.6 % matches
    0.756). Identifiers with underscores (run names, dataset keys, metric names)
    must appear verbatim somewhere in the facts. Small integers (up to 10) pass:
    they are counts in prose ("two models", "7 classes").
    """
    known = _numbers_in(facts) + _numbers_in(question)
    unsupported = []
    for raw, pct in _NUMBER.findall(answer):
        value = float(raw.replace(",", "."))
        decimals = len(raw.replace(",", ".").partition(".")[2])
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not decimals and value <= 10:
            continue
        candidates = [value] + ([value / 100] if pct or decimals else [])
        tols = [tol] + ([tol / 100] if pct or decimals else [])
        if not any(abs(f - c) <= t for f in known for c, t in zip(candidates, tols)):
            unsupported.append(f"{raw}{pct}")
    blob = json.dumps(facts, ensure_ascii=False) + " " + question
    for ident in set(_IDENT.findall(answer)):
        if ident not in blob:
            unsupported.append(ident)
    return sorted(set(unsupported))


def build_models_graph(llm=None, llm_name: str = "", device: str = "cpu"):
    def collect_facts(state: ModelsState) -> dict:
        try:
            facts, charts = collect(state.get("request", ""), state.get("min_balanced_acc", 0.0), device)
        except Exception as exc:
            return {"error": f"could not collect the model facts: {exc}"}
        return {"facts": facts, "charts": charts}

    def explain(state: ModelsState) -> dict:
        if state.get("error"):
            return {}
        if llm is None:
            return {"answer": summary(state["facts"]), "answered_by": "deterministic"}
        try:
            out = llm.invoke([("system", EXPLAIN_SYSTEM),
                              ("human", json.dumps(build_explain_payload(state), ensure_ascii=False))])
            text = out.content if isinstance(out.content, str) else "".join(
                b.get("text", "") for b in out.content if isinstance(b, dict))
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models
        except Exception as exc:
            return {"answer": summary(state["facts"]), "answered_by": "deterministic",
                    "llm_error": str(exc)}
        return {"answer": text, "answered_by": llm_name or type(llm).__name__, "llm_output": text}

    def check(state: ModelsState) -> dict:
        if state.get("error"):
            return {"report": state["error"]}
        if state.get("answered_by") == "deterministic":
            claims = {"verdict": "not_checked", "unsupported": []}
            answer = state["answer"]
            if state.get("llm_error"):
                answer = "The LLM could not answer, so this is the summary computed in code.\n\n" + answer
        else:
            bad = check_claims(state["answer"], state["facts"], state.get("request", ""))
            claims = {"verdict": "rejected" if bad else "supported", "unsupported": bad}
            answer = state["answer"] if not bad else (
                "The LLM's answer cited figures or names that are not in the facts "
                f"({', '.join(bad)}), so it is not shown. This is the summary computed in code.\n\n"
                + summary(state["facts"]))
        return {"claims": claims, "report": answer,
                "messages": [AIMessage(content=answer, name=AGENT_NAME)]}

    g = StateGraph(ModelsState)
    g.add_node("collect_facts", collect_facts)
    g.add_node("explain", explain)
    g.add_node("check_claims", check)
    g.add_edge(START, "collect_facts")
    g.add_edge("collect_facts", "explain")
    g.add_edge("explain", "check_claims")
    g.add_edge("check_claims", END)
    return g.compile(name=AGENT_NAME)


def build_models_agent(model: Optional[str] = None, device: str = "cpu"):
    """`model`: "provider:model" for the answer; None = the deterministic summary only."""
    llm = None
    if model:
        from tester.agent import build_reasoning_llm
        llm = build_reasoning_llm(model)
    return build_models_graph(llm, llm_name=model or "", device=device)
