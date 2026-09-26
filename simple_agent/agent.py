#!/usr/bin/env python3
"""
A minimal LangGraph agent that can train FPViT on DermaMNIST.

It is deliberately small: one prebuilt ReAct agent, three tools, one system
prompt. Everything the agent knows how to *do* lives in `fpvit_tools.py`, which
knows nothing about LangGraph - so the harness can grow (memory, checkpointing,
human-in-the-loop, more tools) without touching the pipeline, and the pipeline
(`train.py`, `fpvit/`) is never modified at all.

Usage:

    export ANTHROPIC_API_KEY=...            # or: ant auth login
    pip install -r simple_agent/requirements.txt

    # one-shot
    python -m simple_agent.agent "Train for 2 epochs with adamw at lr 3e-4 and tell me what happened"

    # chat (conversation kept in memory for the length of the process)
    python -m simple_agent.agent --interactive

Any provider that supports tool calling works - see build_llm():

    python -m simple_agent.agent --model ollama:qwen3.6 --check
    python -m simple_agent.agent --model openrouter:qwen/qwen3-235b-a22b "..."

Where to plug things in later:

    build_agent(checkpointer=...)   short-term memory / resumable threads
                                    (InMemorySaver now, SqliteSaver or Postgres
                                    later - same interface)
    build_agent(store=...)          long-term memory across threads
                                    (langgraph.store.memory.InMemoryStore, then
                                    a real store)
    build_agent(extra_tools=[...])  more tools (evaluation, prediction, a paper
                                    search, ...) without touching this file's logic
    create_react_agent(..., pre_model_hook=...)  context trimming / summarisation
"""

from __future__ import annotations

import argparse
import os
import sys

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

try:                                      # langgraph >= 0.6 keeps both paths
    from langgraph.prebuilt import create_react_agent
except ImportError:                        # langchain 1.x layout
    from langchain.agents import create_agent as create_react_agent

from simple_agent import fpvit_tools


DEFAULT_MODEL = "claude-opus-5"


SYSTEM_PROMPT = f"""\
You are a training agent for FPViT (Feature Pyramid Vision Transformer) on
DermaMNIST, a 7-class dermatoscopic image dataset. You run training jobs by
calling tools, then report what actually happened.

What you must know about this dataset and model:

- The train split is heavily imbalanced: {fpvit_tools.TRAIN_CLASS_COUNTS}
  for classes {list(range(7))} = {fpvit_tools.CLASS_NAMES}.
  Class 5 (melanocytic nevi) alone is 67% of the data, so a model that always
  predicts class 5 already scores 0.669 accuracy. NEVER judge a run by plain
  accuracy. Use balanced accuracy, macro-F1, macro-AUC and the per-class recall
  the tool returns; a class with recall 0.0 has collapsed and that is the
  headline finding, whatever the accuracy says.
- One epoch costs roughly 200 seconds on Apple Silicon (MPS). Always pass a
  small `epochs` and rely on `max_seconds`; the run stops cleanly at the budget
  and still reports every epoch it completed.
- `optimizer` is 'sgd' (the paper's recipe, lr 1e-3) or 'adamw' (usually a
  stronger default for the ViT heads, lr around 3e-4).
- `class_weight` ('inverse' / 'effective') and `balanced_sampler` are the two
  levers against the imbalance. They are alternatives, not complements.
- The test split is off limits: training and model selection use train/val only.

How to work:

1. Turn the user's request into one concrete configuration. If they did not say
   how long to train, choose a short run (2-3 epochs) and say so - do not ask
   the user for a number you can reasonably pick yourself.
2. Call `train_fpvit` once. It returns the whole outcome; do not call it again
   with the same configuration.
3. Read the result and report: what was run, the metrics at the selected epoch,
   which classes collapsed, whether it was still improving or already
   overfitting (compare the train/val gap and the loss curve), and one concrete
   next step. Be brief and concrete; quote the numbers you actually received.
4. If a tool returns status 'invalid_config', fix the named fields and retry.
   If it returns 'error', report the error - do not invent results.

Never report a metric that did not come back from a tool call.
"""


# --- the tools the model sees ----------------------------------------------
# Thin wrappers: the docstring and the signature are the whole API contract the
# LLM reads, the behaviour lives in fpvit_tools.

@tool
def train_fpvit(epochs: int = 3,
                lr: float = 1e-3,
                optimizer: str = "sgd",
                batch_size: int = 128,
                aug_preset: str = "default",
                class_weight: str = "none",
                balanced_sampler: bool = False,
                select_on: str = "macro_auc",
                max_seconds: float = 1800.0,
                seed: int = 42,
                run_name: str = "") -> dict:
    """Train FPViT on DermaMNIST and return the outcome of the run.

    Args:
        epochs: number of epochs, 1..100. One epoch is ~200 s on MPS.
        lr: learning rate (paper: 1e-3 with sgd; try ~3e-4 with adamw).
        optimizer: 'sgd' or 'adamw'.
        batch_size: 8..512, default 128.
        aug_preset: 'none' | 'dihedral' | 'default' | 'strong' | 'paper'.
        class_weight: 'none' | 'inverse' | 'effective' - per-class weights in
            the training loss, against the class imbalance.
        balanced_sampler: resample each epoch to be class-balanced. An
            alternative to class_weight, not a complement.
        select_on: validation metric that picks the best checkpoint -
            'macro_auc' | 'balanced_acc' | 'macro_f1' | 'acc' | 'loss'.
        max_seconds: wall-clock budget for this run. The run stops cleanly
            between epochs when the budget is spent and still reports metrics.
        seed: random seed.
        run_name: short name for the run directory (letters, digits, _ and -).

    Returns:
        A dict with 'status' ('ok' | 'partial' | 'error' | 'invalid_config'),
        the run directory, and a 'summary' holding the metrics at the selected
        epoch, per-class recall, collapsed classes, the loss curve and timings.
    """
    return fpvit_tools.train_fpvit(
        epochs=epochs, lr=lr, optimizer=optimizer, batch_size=batch_size,
        aug_preset=aug_preset, class_weight=class_weight,
        balanced_sampler=balanced_sampler, select_on=select_on,
        max_seconds=max_seconds, seed=seed, run_name=run_name,
    )


@tool
def list_runs() -> dict:
    """List the training runs launched so far, newest first, with their scores."""
    return fpvit_tools.list_runs()


@tool
def get_run_summary(run_name: str) -> dict:
    """Read back the full summary of one previous run, by its run_name."""
    return fpvit_tools.get_run_summary(run_name)


TOOLS = [train_fpvit, list_runs, get_run_summary]


# --- the agent --------------------------------------------------------------

def build_llm(model: str = DEFAULT_MODEL, **overrides):
    """Build the chat model from a "provider:model" string.

        anthropic:claude-opus-5          (the default; the prefix is optional)
        ollama:qwen3.5:4b                local, via a running `ollama serve`
        openrouter:qwen/qwen3-235b-a22b  any OpenRouter model id

    Only the first colon splits provider from model id, because Ollama tags are
    themselves colon-separated ("qwen3.5:4b").

    Whatever the provider, the model must support **tool calling**: this agent
    does nothing except call tools. Ollama reports that per model
    (`ollama show <model>` -> Capabilities: tools); a model without it will
    chat politely and never launch a run. Use `--check` to find out in two
    seconds instead of after a training.
    """
    provider, _, model_id = model.partition(":")
    if not model_id:                      # no prefix -> Anthropic, as before
        provider, model_id = "anthropic", model

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {
            "model": model_id,
            "max_tokens": 8000,
            # Adaptive thinking: the model decides how much to reason per turn.
            # Anthropic-only, hence its place in this branch.
            "thinking": {"type": "adaptive"},
        }
        return ChatAnthropic(**{**kwargs, **overrides})

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {
            "model": model_id,
            "temperature": 0.0,
            # Ollama's own default context is small (2048 tokens on older
            # builds) and it truncates SILENTLY: the system prompt plus one
            # tool result would fall off the front and the model would answer
            # from nothing. Ask for a real window explicitly.
            "num_ctx": 8192,
            "validate_model_on_init": True,   # fail now, not mid-conversation
        }
        return ChatOllama(**{**kwargs, **overrides})

    if provider in ("openrouter", "openai"):
        from langchain_openai import ChatOpenAI
        if provider == "openrouter":
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            kwargs = {"model": model_id, "base_url": base_url, "api_key": api_key,
                      "temperature": 0.0}
        else:
            kwargs = {"model": model_id, "temperature": 0.0}
        return ChatOpenAI(**{**kwargs, **overrides})

    raise ValueError(f"unknown provider {provider!r} in {model!r}; "
                     "use anthropic:, ollama:, openrouter: or openai:")


def build_agent(model: str = DEFAULT_MODEL,
                checkpointer=None,
                store=None,
                extra_tools: list | None = None,
                system_prompt: str = SYSTEM_PROMPT,
                **model_kwargs):
    """Build the ReAct agent.

    Args:
        model: "provider:model" string - see build_llm (default: claude-opus-5).
        checkpointer: LangGraph checkpointer for short-term memory. Pass
            InMemorySaver() for a conversation that survives within one process,
            or a SqliteSaver/PostgresSaver later for one that survives restarts.
            With a checkpointer set, every call needs
            config={"configurable": {"thread_id": "..."}}.
        store: LangGraph store for long-term memory across threads.
        extra_tools: additional tools to hand the model.
        model_kwargs: passed through to the chat model constructor.
    """
    llm = build_llm(model, **model_kwargs)
    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store

    return create_react_agent(
        llm,
        TOOLS + list(extra_tools or []),
        prompt=system_prompt,
        **kwargs,
    )


def _print_new_messages(chunk, seen: set) -> None:
    """Print tool calls and assistant text as they appear, once each."""
    for message in chunk.get("messages", []):
        key = getattr(message, "id", None) or id(message)
        if key in seen:
            continue
        seen.add(key)
        kind = getattr(message, "type", "")
        if kind == "ai":
            for call in getattr(message, "tool_calls", []) or []:
                print(f"\n  -> {call['name']}({call['args']})\n", flush=True)
            # langchain-core 1.x exposes .text as a property; older versions
            # as a method. The 1.x value is a str subclass, hence the isinstance.
            raw = getattr(message, "text", "")
            text = raw() if (callable(raw) and not isinstance(raw, str)) else str(raw)
            if text.strip():
                print(text.strip(), flush=True)
        elif kind == "tool":
            preview = str(message.content)
            if len(preview) > 400:
                preview = preview[:400] + " ...[truncated in this console view]"
            print(f"  <- {message.name}: {preview}\n", flush=True)


def run_once(agent, prompt: str, config: dict | None = None) -> None:
    seen: set = set()
    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]},
                              config=config or {}, stream_mode="values"):
        _print_new_messages(chunk, seen)


def check_tool_calling(model: str) -> int:
    """Does this model actually emit a tool call? Ask it before trusting it.

    Tool calling is the one capability this whole agent rests on, and support
    varies wildly: a local 4B model, or an OpenAI-compatible gateway that
    accepts `tools` and ignores them, will answer in prose instead of calling
    anything - and the failure looks like "the agent decided not to train".
    Two seconds here beats discovering it later.
    """
    print(f"Model: {model}")
    try:
        llm = build_llm(model)
    except Exception as exc:
        print(f"  FAIL: could not build the model: {type(exc).__name__}: {exc}")
        return 1

    try:
        bound = llm.bind_tools(TOOLS)
    except Exception as exc:
        print(f"  FAIL: this model does not accept tools: {type(exc).__name__}: {exc}")
        return 1

    try:
        reply = bound.invoke(
            "List the training runs done so far. Use the tools you have."
        )
    except Exception as exc:
        print(f"  FAIL: the call failed: {type(exc).__name__}: {exc}")
        return 1

    calls = getattr(reply, "tool_calls", None) or []
    if not calls:
        text = str(getattr(reply, "text", "") or reply.content)[:200]
        print("  FAIL: no tool call - the model answered in prose:")
        print(f"        {text!r}")
        print("        (an Ollama model needs the 'tools' capability: "
              "`ollama show <model>`)")
        return 1

    names = [c["name"] for c in calls]
    print(f"  OK: the model called {names}")
    if names != ["list_runs"]:
        print("      note: it picked a different tool than expected, but tool "
              "calling itself works")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple LangGraph training agent for FPViT")
    parser.add_argument("prompt", nargs="*", help="what the agent should do")
    parser.add_argument("--interactive", action="store_true",
                        help="chat loop; the conversation is kept in memory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thread-id", default="default",
                        help="conversation id used by the checkpointer")
    parser.add_argument("--check", action="store_true",
                        help="check that this model can call tools, then exit")
    args = parser.parse_args()

    if args.check:
        return check_tool_calling(args.model)

    if (args.model.split(":")[0] not in ("ollama", "openrouter", "openai")
            and not os.environ.get("ANTHROPIC_API_KEY")
            and not os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Note: ANTHROPIC_API_KEY is not set - the SDK will fall back to an "
              "`ant auth login` profile if you have one.", file=sys.stderr)

    # Short-term memory: swap InMemorySaver for a SqliteSaver to make threads
    # survive a restart. Nothing else in this file has to change.
    agent = build_agent(model=args.model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": args.thread_id}}

    if args.interactive:
        print("FPViT training agent. Ctrl-D or 'exit' to quit.\n")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if line.lower() in {"exit", "quit"}:
                return 0
            if line:
                run_once(agent, line, config)
                print()
        return 0

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        parser.error("give a prompt, or use --interactive")
    run_once(agent, prompt, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
