#!/usr/bin/env python3
"""
CLI of the orchestrator: prediction (testing agent -> reviewer) or training (training agent).

    python run.py orchestrator --train "improve dermatofibroma recognition" --no-llm
    python run.py orchestrator --train --model ollama:qwen3.6:latest "train a resnet18 for 20 minutes"
    python run.py orchestrator "I'd like a better model" --train       # the router extracts the constraints

    python run.py orchestrator input/samples/05_melanoma.png
    python run.py orchestrator input/samples --no-llm                  # vote + automatic checks, no API key
    python run.py orchestrator input/samples --model ollama:qwen3.6 -v # trace + plain-language account
    python run.py orchestrator input/samples --ask-human               # pause on flagged issues
    python run.py orchestrator "what does balanced accuracy measure?" # a question, answered by the LLM
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.types import Command

from orchestrator.graph import build_orchestrator


def _ask(payload: dict) -> dict:
    print("\n" + "=" * 70, file=sys.stderr)
    print(payload["question"], file=sys.stderr)
    for item in payload["flagged"]:
        print(f"\n  {item['file']}: {item['final_class']}", file=sys.stderr)
        for i in item["issues"]:
            print(f"    [{i['severity']}] {i['description']}", file=sys.stderr)
    print("\n  [a] accept   [r] reject   [n] accept with a note", file=sys.stderr)
    choice = ""
    while choice not in {"a", "r", "n"}:
        try:
            print("> ", end="", file=sys.stderr, flush=True)   # prompts on stderr: stdout may be --json
            choice = input().strip().lower()[:1]
        except EOFError:
            choice = "r"                     # no human at the keyboard: do not accept silently
    decision = {"a": "accepted", "r": "rejected", "n": "accepted_with_note"}[choice]
    note = ""
    if choice == "n":
        print("note: ", end="", file=sys.stderr, flush=True)
        try:
            note = input().strip()
        except EOFError:
            pass
    return {"decision": decision, "note": note}


def _answer(payload: dict) -> dict:
    kind = payload.get("kind")
    if kind in (None, "review_flags"):
        return _ask(payload)
    if kind == "clarify_intent":
        print("\n" + payload["question"], file=sys.stderr)
        for o in payload["options"]:
            print(f"  [{o['value'][0]}] {o['label']}", file=sys.stderr)
        print("> ", end="", file=sys.stderr, flush=True)
        try:
            c = input().strip().lower()[:1]
        except EOFError:
            c = "c"
        return {"decision": next((o["value"] for o in payload["options"] if o["value"][0] == c), "cancel")}
    from trainer.agent import answer_interrupt         # the training agent's approvals
    return answer_interrupt(payload)


def main() -> int:
    p = argparse.ArgumentParser(description="DermaAgent orchestrator: prediction + review, or training")
    p.add_argument("inputs", nargs="*", help="image files/folders (prediction) and/or the request in words")
    p.add_argument("--mode", choices=["predict", "train"], default=None,
                   help="skip routing (default: images -> predict, otherwise the router LLM decides)")
    p.add_argument("--train", action="store_true", help="enable the training branch (training agent)")
    p.add_argument("--request", default="", help="optional context for both agents")
    p.add_argument("--model", default="auto",
                   help="LLM of the agents: provider:model, or 'auto' for the default in system/config/llm.json")
    p.add_argument("--reviewer-model", default=None, help="reviewer LLM (default: same as --model)")
    p.add_argument("--no-llm", action="store_true",
                   help="no LLM anywhere: weighted vote + automatic checks")
    p.add_argument("--ask-human", action="store_true",
                   help="pause and ask for a decision when the reviewer flags issues")
    p.add_argument("--min-balanced-acc", type=float, default=0.0)
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="trace both agents to stderr and add a plain-language account; -vv adds LLM payloads")
    p.add_argument("--json", action="store_true", help="structured output on stdout")
    args = p.parse_args()

    model = None if args.no_llm else args.model
    reviewer_model = None if args.no_llm else args.reviewer_model
    from pathlib import Path
    images = [x for x in args.inputs if Path(x).exists()]
    words = " ".join(x for x in args.inputs if not Path(x).exists())
    request = " ".join(t for t in (args.request, words) if t)
    app = build_orchestrator(model=model, reviewer_model=reviewer_model,
                             verbose=args.verbose, ask_human=args.ask_human,
                             predictor_kwargs={"min_balanced_acc": args.min_balanced_acc,
                                               "exclude": args.exclude},
                             with_trainer=args.train or args.mode == "train", with_router=True,
                             trainer_unavailable="Training is not enabled: add --train to the command.")
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    out = app.invoke({"image_paths": images, "request": request, "mode": args.mode or ""}, config)
    while out.get("__interrupt__"):
        out = app.invoke(Command(resume=_answer(out["__interrupt__"][0].value)), config)

    if args.json:
        rv = out.get("review_result", {})
        print(json.dumps({
            "error": out.get("error"),
            "final": (out.get("test_result") or {}).get("final"),
            "review": rv.get("review"), "automatic_checks": rv.get("checks"),
            "decisive_model": rv.get("decisive"),
            "needs_human_review": out.get("needs_human_review"),
            "human_decision": out.get("human_decision"),
        }, indent=1, ensure_ascii=False, default=str))
    else:
        print(out["report"])
    return 1 if out.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
