"""
Tracing of the predictor graph: what went into and came out of each node.

Every node is wrapped, always: the lines it produces are returned in the
`trace` channel of the state (appended, one list per node), so a downstream
agent - the reviewer - reads exactly what a human would have seen with -v.
`verbose` only decides whether the lines are ALSO printed live, to stderr, so
`--json` on stdout stays machine-readable.

The tracer wraps the node functions themselves rather than the CLI's stream
loop, so it works the same whether the graph is run from the CLI, embedded as
a node of a supervisor graph, or called through `as_tool()`.

    verbose=0  collected in state only
    verbose=1  also printed: every node's inputs/outputs and timing, the LLM's
               full structured answer and any override refused in code
    verbose=2  also the complete JSON payload sent to the LLM (printed only,
               never put in the trace: the reviewer gets it structured)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Callable

from predict_agent.models import CLASS_NAMES


def _short(name: str) -> str:
    return name.split(" / ")[0].split(" (")[0]


def _top(probs: list[float], k: int = 3) -> str:
    idx = sorted(range(len(probs)), key=lambda c: probs[c], reverse=True)[:k]
    return ", ".join(f"{_short(CLASS_NAMES[c])} {probs[c]:.2f}" for c in idx)


class NodeTracer:
    def __init__(self, verbose: int = 0, extra: dict | None = None, stream=None):
        self.verbose = verbose
        self.extra = extra or {}
        self.stream = stream or sys.stderr

    def _print(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    def wrap(self, name: str, fn: Callable) -> Callable:
        printer = getattr(self, f"_print_{name}", None)

        def traced(state):
            lines: list[str] = []
            emit = lines.append
            emit(f"=== [{name}] " + "=" * max(0, 60 - len(name)))
            t0 = time.perf_counter()
            update = fn(state)
            dt = time.perf_counter() - t0
            if update.get("error"):
                emit(f"  ERROR: {update['error']}")
            elif printer:
                printer(emit, state, update)
            emit(f"  ({dt:.2f}s)")
            if self.verbose:
                self._print("\n" + "\n".join(lines))
            return {**update, "trace": lines}

        traced.__name__ = name
        return traced

    # --- one printer per node ----------------------------------------------

    def _print_load_inputs(self, emit, state, up):
        if not state.get("image_paths"):
            emit(f"  paths taken from the last human message: {up.get('request', '')!r}")
        for img in up["images"]:
            extra = f"  RESIZED from {img['original_size']}" if img["resized"] else ""
            emit(f"  {img['file']}  sha256={img['sha256'][:12]}  {img['path']}{extra}")

    def _print_recall_memory(self, emit, state, up):
        emit(f"  deterministic memory: {json.dumps(up['log_stats'])}")
        for img in state["images"]:
            past = up["past"].get(img["sha256"], [])
            if not past:
                emit(f"  {img['file']}: never seen before")
            for r in past:
                emit(f"  {img['file']}: seen {r['timestamp']} -> {r['final_class']} "
                     f"(vote {r['vote_class']}, {r['confidence_level']}, models {r['models']})")

    def _print_run_models(self, emit, state, up):
        emit("  ensemble:")
        for m in up["models"]:
            emit(f"    {m['name']}: epoch {m['epoch']}, val bal.acc "
                 f"{m['val_balanced_acc']:.3f} -> skill {m['skill']:.3f}, "
                 f"blind to {m['blind_classes'] or '-'}  [{m['optimizer']}, "
                 f"class_weight={m['class_weight']}]")
        for i, img in enumerate(state["images"]):
            emit(f"  {img['file']}:")
            for name, probs in up["predictions"].items():
                emit(f"    {name:<28} {_top(probs[i])}")

    def _print_vote(self, emit, state, up):
        for img, v in zip(state["images"], up["votes"]):
            mc = v["most_confident_model"]
            emit(f"  {img['file']}:")
            emit("    soft vote : " + ", ".join(f"{_short(k)} {p:.2f}"
                                               for k, p in list(v["soft_vote"].items())[:4]))
            emit("    hard vote : " + ", ".join(f"{_short(k)} {w:.2f}"
                                               for k, w in v["hard_vote"].items()))
            emit(f"    most conf.: {mc['model']} -> {_short(mc['predicted_class'])} "
                 f"({mc['confidence']:.2f})")
            emit(f"    => {v['vote_class']} p={v['vote_probability']:.2f}, margin "
                 f"{v['margin']:.2f}, agree {v['models_agreeing_with_vote']}, "
                 f"level {v['confidence_level'].upper()}")
            emit(f"    candidates {v['candidate_class_ids']}; blind to vote class: "
                 f"{v['models_blind_to_vote_class'] or '-'}")

    def _print_retrieve_knowledge(self, emit, state, up):
        shown = set()
        for img, ev in zip(state["images"], up["evidence"]):
            emit(f"  {img['file']}: {len(ev)} chunk(s) {[c['chunk_id'] for c in ev]}")
            for c in ev:
                if c["chunk_id"] in shown:
                    continue
                shown.add(c["chunk_id"])
                emit(f"    [{c['chunk_id']}] score {c['score']:.3f}  about: {c['about_class']}")
                emit(f"      {c['title']} ({c['citation']})")
                emit(f"      \"{c['text'][:220]}...\"")

    def _print_reason(self, emit, state, up):
        payload, raw = up.get("llm_input"), up.get("llm_output")
        if not payload:
            emit("  no LLM: the weighted vote is the final answer")
        else:
            emit(f"  LLM: {up['reasoner'] if raw else '(failed)'}; payload "
                 f"{len(json.dumps(payload, ensure_ascii=False)):,} chars")
            if self.verbose >= 2:
                self._print("  --- payload sent to the LLM ---\n"
                            + json.dumps(payload, ensure_ascii=False, indent=1)
                            + "\n  --- end payload ---")
        if up.get("llm_error"):
            emit(f"  LLM ERROR, vote kept: {up['llm_error']}")
        if raw:
            emit("  --- LLM answer (raw, before validation) ---")
            for d in raw.get("decisions", []):
                emit(f"  {d['file']}: class {d['final_class_id']} "
                     f"({CLASS_NAMES[d['final_class_id']]}), {d['confidence_level']}, "
                     f"overridden={d['overridden']}")
                emit(f"    rationale: {d['rationale']}")
                emit(f"    knowledge used: {d.get('knowledge_used') or '-'}")
                emit(f"    next step: {d.get('suggested_next_step')}")
            emit(f"  summary: {raw.get('summary')}")
        for r in up.get("rejected_overrides", []):
            emit(f"  REFUSED: {r['file']} -> class {r['proposed_class_id']} is not among "
                 f"candidates {r['candidates']}; vote kept")
        for f in up["final"]:
            emit(f"  FINAL {f['file']}: {f['final_class']} ({f['confidence_level']})"
                 + (" [override]" if f["overridden"] else ""))

    def _print_write_log(self, emit, state, up):
        emit(f"  appended {len(state['final'])} line(s), execution_id "
             f"{up['execution_id']} -> {self.extra.get('write_log', {}).get('log_path')}")
