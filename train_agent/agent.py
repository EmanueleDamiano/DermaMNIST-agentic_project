#!/usr/bin/env python3
"""
Entry points for the training agent: builder and CLI.

    python -m train_agent "improve dermatofibroma recognition" --minutes 30
    python -m train_agent --arch resnet18 --trials 3 --minutes 20 --no-llm -v
    python -m train_agent --model ollama:qwen3.6:latest --autonomy supervised

The CLI stops at every human checkpoint (plan, proposals the gate sends to a
human, promotion) and asks on stdin; with stdin closed it answers "reject",
never "approve" - nothing is trained or promoted silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fpvit.zoo import ARCHITECTURES
from train_agent.graph import KB_PATH, build_trainer_graph
from train_agent.space import AUTONOMY, SELECT_ON

DEFAULT_MODEL = "claude-sonnet-5"


def build_trainer(model: str | None = DEFAULT_MODEL, llm=None, verbose: int = 0,
                  checkpointer=None, kb_path=KB_PATH):
    """Compiled training agent. model=None (and llm=None): deterministic policy, no API key.

    Pass a checkpointer when running it standalone; as a subgraph of the
    orchestrator it inherits the parent's.
    """
    roles = None
    if llm is None and model:
        from predict_agent.agent import build_reasoning_llm
        llm = build_reasoning_llm(model)
    if llm is not None:
        from train_agent.llm import LLMRoles
        roles = LLMRoles(llm, name=model or "")
    return build_trainer_graph(roles=roles, kb_path=kb_path, checkpointer=checkpointer, verbose=verbose)


def _ask(prompt: str) -> str:
    print(prompt, end="", file=sys.stderr, flush=True)
    try:
        return input().strip()
    except EOFError:
        return ""


def answer_interrupt(payload: dict) -> dict:
    kind = payload.get("kind")
    err = sys.stderr
    print("\n" + "=" * 72, file=err)
    print(payload.get("question", ""), file=err)
    if kind == "approve_plan":
        for k, v in payload["plan"].items():
            print(f"  {k}: {v}", file=err)
        print(f"  (editable: {', '.join(payload['editable'])})", file=err)
    elif kind == "approve_proposal":
        pr = payload["proposal"]
        print(f"  action: {pr['action']}  arch: {pr['arch']}  parent: {pr.get('parent_run')}", file=err)
        print(f"  hyperparameters: {pr['hparams']}", file=err)
        print(f"  rationale: {pr['rationale']}", file=err)
        print(f"  augmentation: {payload['augmentation_resolved']}", file=err)
        print(f"  why: {pr['augmentation_rationale']}", file=err)
        print(f"  changes: {payload['changes']}", file=err)
        for r in payload["why_human"]:
            print(f"  → {r}", file=err)
        for f in payload["findings"]:
            print(f"  [{f['severity']}] {f['description']}", file=err)
    elif kind == "approve_promotion":
        ev = payload["evaluation"]
        print(f"  candidate {payload['candidate']['run']}: ensemble {ev['ensemble_now']['balanced_acc']:.4f} → "
              f"{ev['ensemble_with_candidate']['balanced_acc']:.4f} ({ev['delta_balanced_acc']:+.4f}); "
              f"recommendation: {payload['recommendation']}", file=err)
    opts = payload.get("options", [])
    print("  " + "   ".join(f"[{o['value'][0]}] {o['label']}" for o in opts), file=err)
    choice = _ask("> ").lower()[:1]
    decision = next((o["value"] for o in opts if o["value"][0] == choice), "reject")
    out = {"decision": decision, "note": _ask("note (enter to skip): ")}
    if kind == "approve_plan" and decision == "approve":
        raw = _ask("edits as key=value separated by spaces (enter for none): ")
        out["edits"] = dict(kv.split("=", 1) for kv in raw.split() if "=" in kv)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="DermaAgent training agent (LangGraph)")
    p.add_argument("request", nargs="*", help="what you want from this campaign, in words")
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), default=None,
                   help="fix the architecture (default: the agent chooses; FPViT unless it argues otherwise)")
    p.add_argument("--trials", type=int, default=None, help="max training segments")
    p.add_argument("--minutes", type=float, default=None, help="wall-clock budget of the campaign")
    p.add_argument("--epochs-per-run", type=int, default=None)
    p.add_argument("--patience", type=int, default=None, help="epochs without improvement that wake the agent")
    p.add_argument("--autonomy", choices=AUTONOMY, default=None)
    p.add_argument("--select-on", choices=SELECT_ON, default=None)
    p.add_argument("--target", type=float, default=None, help="stop when the selection metric reaches this")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LLM for proposer and analyst, provider:model")
    p.add_argument("--no-llm", action="store_true", help="deterministic policy instead of an LLM")
    p.add_argument("--yes", action="store_true", help="approve the plan without asking (proposals and "
                                                      "promotion still follow the autonomy rules)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()

    constraints = {"arch": args.arch, "max_trials": args.trials, "max_minutes": args.minutes,
                   "max_epochs_per_run": args.epochs_per_run, "patience": args.patience,
                   "autonomy": args.autonomy, "select_on": args.select_on, "target_score": args.target,
                   "auto_approve_plan": args.yes}
    app = build_trainer(None if args.no_llm else args.model, verbose=max(args.verbose, 1),
                        checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    out = app.invoke({"request": " ".join(args.request), "constraints": constraints}, config)
    while out.get("__interrupt__"):
        out = app.invoke(Command(resume=answer_interrupt(out["__interrupt__"][0].value)), config)
    print(out.get("report", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
