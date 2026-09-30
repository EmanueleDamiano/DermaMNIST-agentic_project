#!/usr/bin/env python3
"""
Entry points for the prediction agent: builder, supervisor tool, CLI.

    # CLI - one or more images and/or folders
    python run.py tester input/samples/05_melanoma.png input/samples/01_actinic_keratoses.png
    python run.py tester input/samples --no-llm          # vote only, no API key
    python run.py tester input/samples --model ollama:qwen3.6
    python run.py tester input/samples --model anthropic:claude-opus-5-5
    python run.py tester input/samples --no-llm -v       # trace every node

    # in Python
    from tester import build_predictor
    predictor = build_predictor(model="auto")     # default model of system/config/llm.json, if any
    out = predictor.invoke({"image_paths": ["input/samples/05_melanoma.png"]})
    out["final"]        # structured, one dict per image
    out["report"]       # the same, as text

Multi-agent: see `as_tool()` (supervisor calls it like any tool) and the README
(compiled graph added as a node of a `MessagesState` supervisor graph).
"""

from __future__ import annotations

import argparse
import json
import sys

from tester.graph import AGENT_NAME, build_predictor_graph
from tester.memory import DEFAULT_CORPUS_PATH, DEFAULT_LOG_PATH, ExecutionLog, KnowledgeBase
from tester.models import ModelZoo

DEFAULT_MODEL = "auto"


def build_reasoning_llm(model: str | None, **overrides):
    """Chat model for a "provider:model" address (see llm/registry.py).

    "auto" means the default model of system/config/llm.json; None, "" or "none" means
    no LLM, and the agent falls back to its deterministic rules.
    """
    from llm import build_chat_model, resolve_model
    spec = resolve_model(model)
    return build_chat_model(spec, **overrides) if spec else None


def build_predictor(model: str | None = DEFAULT_MODEL,
                    llm=None,
                    model_roots=None,
                    corpus_path=DEFAULT_CORPUS_PATH,
                    log_path=DEFAULT_LOG_PATH,
                    device: str = "auto",
                    min_balanced_acc: float = 0.0,
                    exclude=(),
                    checkpointer=None,
                    verbose: int = 0):
    """Build the compiled prediction graph.

    Args:
        model: "provider:model" for the reasoning step, or None for a fully
            deterministic agent (the vote is the answer, no API key needed).
        llm: an already-built LangChain chat model; takes precedence over `model`
            (a supervisor can share its own model instance).
        model_roots: folders whose `*/best_model.pt` form the ensemble
            (default: model/baseline/, model/promoted/, output/runs/).
        corpus_path: knowledge base for retrieval; None disables it.
        log_path: JSONL execution log (deterministic memory).
        min_balanced_acc: drop checkpoints below this val balanced accuracy.
        exclude: run names to leave out of the ensemble.
        checkpointer: LangGraph checkpointer, if the caller wants thread state.
        verbose: 0 silent; 1 print every node's inputs/outputs and the LLM's
            full answer to stderr; 2 also the complete payload sent to the LLM.
    """
    if llm is None and model:
        from llm import resolve_model
        model = resolve_model(model)
        llm = build_reasoning_llm(model) if model else None
    zoo = ModelZoo(model_roots, device=device, min_balanced_acc=min_balanced_acc,
                   exclude=exclude)
    kb = KnowledgeBase(corpus_path) if corpus_path else None
    return build_predictor_graph(zoo, ExecutionLog(log_path), kb, llm=llm,
                                 llm_name=model or type(llm).__name__ if llm else "",
                                 checkpointer=checkpointer, verbose=verbose)


def as_tool(predictor=None, **build_kwargs):
    """The predictor as a LangChain tool, for a supervisor / ReAct orchestrator.

    The tool returns the structured `final` list plus the text report, so the
    calling agent gets both the numbers and something to show the user.
    """
    from langchain_core.tools import tool

    graph = predictor or build_predictor(**build_kwargs)

    @tool(AGENT_NAME)
    def classify_skin_lesions(image_paths: list[str], request: str = "") -> dict:
        """Classify dermatoscopic images with the local FPViT ensemble (DermaMNIST, 7 classes).

        Runs every local checkpoint, combines them by weighted voting, checks the
        log of past predictions for the same images, retrieves literature on the
        leading classes, and returns a final class, a confidence level and a
        rationale per image. Research use only - not a diagnosis.

        Args:
            image_paths: image files and/or folders of images.
            request: optional context from the user (language, what matters).
        """
        out = graph.invoke({"image_paths": image_paths, "request": request})
        if out.get("error"):
            return {"error": out["error"]}
        return {"predictions": [{k: f[k] for k in (
                    "file", "final_class_id", "final_class", "confidence_level",
                    "overridden", "vote_class", "vote_probability", "margin",
                    "rationale", "suggested_next_step", "sources")} for f in out["final"]],
                "report": out["report"]}

    return classify_skin_lesions


def main() -> int:
    p = argparse.ArgumentParser(description="FPViT ensemble prediction agent")
    p.add_argument("images", nargs="+", help="image files and/or folders")
    p.add_argument("--request", default="", help="optional context for the reasoning step")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="reasoning LLM as provider:model, 'auto' for the configured default (python run.py llm)")
    p.add_argument("--no-llm", action="store_true",
                   help="deterministic: the weighted vote is the final answer")
    p.add_argument("--models-root", action="append", default=None,
                   help="folder containing <run>/best_model.pt; repeatable")
    p.add_argument("--exclude", action="append", default=[], help="run name to skip; repeatable")
    p.add_argument("--min-balanced-acc", type=float, default=0.0)
    p.add_argument("--no-kb", action="store_true", help="disable knowledge-base retrieval")
    p.add_argument("--device", default="auto")
    p.add_argument("--json", action="store_true", help="print the structured result as JSON")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="trace every node to stderr; -vv also prints the full LLM payload")
    args = p.parse_args()

    predictor = build_predictor(
        model=None if args.no_llm else args.model,
        model_roots=args.models_root,
        corpus_path=None if args.no_kb else DEFAULT_CORPUS_PATH,
        device=args.device, min_balanced_acc=args.min_balanced_acc, exclude=args.exclude,
        verbose=args.verbose,
    )
    out = predictor.invoke({"image_paths": args.images, "request": args.request})
    if args.json:
        print(json.dumps({k: out.get(k) for k in ("final", "models", "reasoner", "error")},
                         indent=1, ensure_ascii=False, default=str))
    else:
        print(out["report"])
    return 1 if out.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
