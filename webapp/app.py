"""
DermaAgent platform: the orchestrator (testing agent -> reviewer -> human) with
a live view of which agent is working and what it is doing.

    python -m webapp                 # http://127.0.0.1:8000
    python -m webapp --port 8080

Standard library HTTP server, one page (`page.html`), jobs in background
threads, the browser polls `/api/status`. Nothing in the agents was written
for this UI: the live graph comes from LangGraph's own task stream
(`stream_mode="tasks", subgraphs=True`), which reports the start and the result
of every node - the orchestrator's, and those of the tester and reviewer
graphs running inside it, each tagged with its namespace. The human-in-the-loop
pause is the orchestrator's `interrupt()`; the page answers it and the job is
resumed with `Command(resume=...)`.

Only images uploaded through the page or the files in `test_samples/` can be
analysed: no path from the browser ever reaches the filesystem.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
import time
import traceback
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from PIL import Image

from predict_agent.models import CLASS_NAMES, IMAGE_SUFFIXES, PROJECT_ROOT, ModelZoo

PAGE_PATH = Path(__file__).with_name("page.html")
PAGES = {"/": PAGE_PATH, "/context": Path(__file__).with_name("context.html"),
         "/kb": Path(__file__).with_name("kb.html")}
SAMPLES_DIR = PROJECT_ROOT / "test_samples"
UPLOAD_ROOT = PROJECT_ROOT / "runs_predict" / "uploads"
PREDICTIONS_LOG = PROJECT_ROOT / "runs_predict" / "predictions_log.jsonl"
REVIEWS_LOG = PROJECT_ROOT / "runs_predict" / "reviews_log.jsonl"
CORPUS = PROJECT_ROOT / "corpus.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 16
MODEL_RE = re.compile(r"^(?:(?:anthropic|ollama|openrouter|openai):)?[\w.\-:/]{1,120}$")

TESTER_STEPS = ["load_inputs", "recall_memory", "run_models", "vote",
                "retrieve_knowledge", "reason", "write_log"]
REVIEWER_STEPS = ["gather", "review", "render"]
STEP_LABELS = {
    "load_inputs": "Reads the images", "recall_memory": "Checks the deterministic memory",
    "run_models": "Runs the local models", "vote": "Combines the votes",
    "retrieve_knowledge": "Searches the knowledge base", "reason": "Reasons about the decision",
    "write_log": "Writes the log", "gather": "Automatic checks and independent KB",
    "review": "Checks consistency", "render": "Writes the report",
    "finalize": "Saves the review", "human_review": "Waits for the human decision",
}
ACTORS = {"orchestrator": "Orchestrator", "tester": "Testing agent",
          "reviewer": "Reviewer agent", "trainer": "Training agent", "human": "Human"}

# Training agent nodes shown in the graph (give_up is folded into validate).
TRAINER_STEPS = ["plan", "approve_plan", "propose", "validate", "review_proposal", "approve_proposal",
                 "train", "analyse", "decide", "evaluate", "approve_promotion", "promote", "finish"]
STEP_LABELS.update({
    "route": "Routes the request", "clarify": "Asks what you want to do", "answer": "Answers",
    "trainer": "Hands over to the Training agent",
    "plan": "Prepares the plan", "approve_plan": "Waits for plan approval",
    "propose": "Proposes the next run", "validate": "Validates the proposal",
    "review_proposal": "Reviewer and autonomy gate", "approve_proposal": "Waits for proposal approval",
    "train": "Trains (stops at plateau)", "analyse": "Analyses the run", "decide": "Decides whether to continue",
    "evaluate": "Evaluates the ensemble with the candidate", "approve_promotion": "Waits for the promotion decision",
    "promote": "Promotes into the ensemble",
})

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()          # one prediction pipeline at a time: models and LLMs are shared
TRAIN_LOCK = threading.Lock()        # one training campaign at a time; predictions do not wait for it
APPS: dict[str, Any] = {}
APPS_LOCK = threading.Lock()
KB: dict[str, Any] = {}
KB_LOCK = threading.Lock()
HEALTH: dict[str, Any] = {"time": 0.0}


# ---------------------------------------------------------------------------
# Environment / health
# ---------------------------------------------------------------------------

def _ollama_models() -> list[str]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=0.8) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _archs() -> dict:
    from fpvit.zoo import ARCHITECTURES
    return ARCHITECTURES


def health(force: bool = False) -> dict:
    """Cached for 30 s: scanning the checkpoints loads them from disk."""
    if not force and time.time() - HEALTH["time"] < 30:
        return HEALTH["data"]
    try:
        zoo = ModelZoo()
        models = [{"name": c.name, "epoch": c.epoch, "val_balanced_acc": round(c.val_balanced_acc, 3),
                   "skill": round(c.skill, 3), "blind_classes": c.blind_classes}
                  for c in zoo.cards.values()]
    except Exception as exc:
        models = []
        print(f"health: could not scan the models: {exc}", flush=True)
    try:
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        kb = {"ready": True, "sources": len(corpus.get("sources", [])),
              "chunks": sum(len(s.get("chunks", [])) for s in corpus.get("sources", []))}
    except Exception:
        kb = {"ready": False, "sources": 0, "chunks": 0}
    ollama = _ollama_models()
    anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    llm_options = ([f"ollama:{m}" for m in ollama]
                   + (["claude-sonnet-5", "claude-opus-5-5"] if anthropic else [])
                   + (["openrouter:qwen/qwen3-235b-a22b"] if os.environ.get("OPENROUTER_API_KEY") else []))
    default = ("claude-sonnet-5" if anthropic else f"ollama:{ollama[0]}" if ollama else "")
    data = {
        "models": models, "knowledge": kb,
        "llm": {"ollama": ollama, "anthropic": anthropic, "options": llm_options, "default": default},
        "logs": {"predictions": _count_lines(PREDICTIONS_LOG), "reviews": _count_lines(REVIEWS_LOG)},
        "archs": _archs(),
        "samples": sorted(p.name for p in SAMPLES_DIR.iterdir()
                          if p.suffix.lower() in IMAGE_SUFFIXES) if SAMPLES_DIR.is_dir() else [],
        "classes": CLASS_NAMES,
    }
    HEALTH.update(time=time.time(), data=data)
    return data


# ---------------------------------------------------------------------------
# Orchestrators, cached per settings
# ---------------------------------------------------------------------------

def normalize_settings(raw: dict | None) -> dict:
    raw = raw or {}

    def model_field(value, allow_same: bool):
        value = str(value or "").strip()
        if value in ("", "none"):
            return ""
        if allow_same and value == "same":
            return None
        if not MODEL_RE.match(value):
            raise ValueError(f"invalid model name: {value!r}")
        return value

    min_acc = float(raw.get("minBalancedAcc") or 0.0)
    if not 0.0 <= min_acc < 1.0:
        raise ValueError("min balanced accuracy must be between 0 and 1")
    return {
        "tester_model": model_field(raw.get("testerModel"), allow_same=False),
        "reviewer_model": model_field(raw.get("reviewerModel", "same"), allow_same=True),
        "trainer_model": model_field(raw.get("trainerModel", "same"), allow_same=True),
        "explain": bool(raw.get("explain", True)),
        "ask_human": bool(raw.get("askHuman", False)),
        "min_balanced_acc": min_acc,
    }


def get_app(settings: dict, train: bool = False):
    """One compiled orchestrator per distinct configuration, kept for the process life.

    Prediction jobs get exactly the orchestrator they always had; training (and
    free-text) jobs get the one with the training branch and intent routing.
    """
    from orchestrator import build_orchestrator

    key = json.dumps(settings, sort_keys=True) + ("|train" if train else "")
    with APPS_LOCK:
        if key not in APPS:
            extra = {}
            if train:
                tm = settings["trainer_model"]
                extra = {"with_trainer": True,
                         "trainer_model": tm if tm is not None else (settings["tester_model"] or "")}
            APPS[key] = build_orchestrator(
                model=settings["tester_model"] or None,
                reviewer_model=settings["reviewer_model"],
                verbose=0, explain_process=settings["explain"],
                ask_human=settings["ask_human"],
                predictor_kwargs={"min_balanced_acc": settings["min_balanced_acc"]},
                checkpointer=InMemorySaver(),   # needed for get_state() and for interrupt()
                **extra,
            )
        return APPS[key]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def _n(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def _now() -> str:
    return time.strftime("%H:%M:%S")


def update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def activity(job_id: str, actor: str, node: str, message: str, details: list[str] | None = None,
             level: str = "") -> None:
    with JOBS_LOCK:
        items = JOBS[job_id]["activities"]
        items.append({"actor": actor, "node": node, "message": message,
                      "details": details or [], "level": level, "time": _now()})
        del items[:-80]


def _writes(ev: dict) -> dict:
    res = ev.get("result")
    if isinstance(res, list):
        out: dict = {}
        for k, v in res:
            out[k] = v
        return out
    return res or {}


def _summarize(agent: str, node: str, w: dict, job: dict) -> tuple[str, list[str], str]:
    """(message, detail lines, level) for one finished node."""
    files = job.get("_files", [])
    if w.get("error"):
        return f"Error: {w['error']}", [], "error"

    if agent == "tester":
        trace = [ln for ln in w.get("trace", [])[1:] if not ln.strip().startswith("(")]
        if node == "load_inputs":
            job["_files"] = [i["file"] for i in w.get("images", [])]
            return f"Read {_n(len(job['_files']), 'image', 'images')}.", trace, ""
        if node == "recall_memory":
            seen = sum(1 for v in (w.get("past") or {}).values() if v)
            stats = w.get("log_stats") or {}
            return (f"{seen} of {_n(len(files), 'image', 'images')} seen before; the log holds "
                    f"{stats.get('images', 0)} predictions.", trace, "")
        if node == "run_models":
            names = [m["name"] for m in w.get("models", [])]
            return f"Ran {len(names)} local models: {', '.join(names)}.", trace, ""
        if node == "vote":
            parts = [f"{f}: {v['vote_class']} (p={v['vote_probability']:.2f}, {v['confidence_level']})"
                     for f, v in zip(files, w.get("votes", []))]
            return "Weighted vote — " + "; ".join(parts), trace, ""
        if node == "retrieve_knowledge":
            ids = {c["chunk_id"] for ev in w.get("evidence", []) for c in ev}
            return f"Retrieved {len(ids)} passages from the knowledge base.", trace, ""
        if node == "reason":
            who = w.get("reasoner", "")
            parts = [f"{f['file']} → {f['final_class']}" + (" [override]" if f["overridden"] else "")
                     for f in w.get("final", [])]
            level = "warn" if w.get("llm_error") or w.get("rejected_overrides") else ""
            head = "Decision taken by the vote" if who == "deterministic" else f"Decision reasoned by {who}"
            return f"{head}: " + "; ".join(parts), trace, level
        if node == "write_log":
            return f"Predictions saved to the log (execution {w.get('execution_id')}).", trace, ""

    if agent == "reviewer":
        if node == "gather":
            checks = [c for found in w.get("checks", []) for c in found]
            crit = sum(c["severity"] == "critical" for c in checks)
            warn = sum(c["severity"] == "warning" for c in checks)
            dec = {d["model"] for d in w.get("decisive", [])}
            details = []
            for f, found, d, kb in zip(files, w.get("checks", []), w.get("decisive", []),
                                       w.get("reviewer_kb", [])):
                details.append(f"{f}: decisive model {d['model']} "
                               f"({d['share_of_final_class_score']:.0%} of the score)")
                details.append("  independent KB: " + ", ".join(c["chunk_id"] for c in kb))
                details += [f"  [{c['severity']}] {c['description']}" for c in found]
            return (f"Automatic checks: {crit} critical, {warn} warning. "
                    f"Decisive model: {', '.join(sorted(dec))}.", details,
                    "error" if crit else "warn" if warn else "")
        if node == "review":
            rv = w.get("review", {})
            verdicts = [f"{r['file']}: {'consistent' if r['verdict'] == 'consistent' else 'issues'}"
                        for r in rv.get("images", [])]
            details = [f"{r['file']}: [{i['severity']}] {i['description']} — {i.get('evidence', '')}"
                       for r in rv.get("images", []) for i in r.get("issues", [])]
            if w.get("llm_error"):
                details.append(f"LLM error: {w['llm_error']}")
            who = w.get("reviewer", "")
            head = "Review from the automatic checks only" if who == "deterministic" else f"Review by {who}"
            return f"{head} — " + "; ".join(verdicts), details, "warn" if details else ""
        if node == "render":
            flag = w.get("needs_human_review")
            return ("Report ready: " + ("needs human review." if flag
                                         else "no contradiction found."),
                    [], "warn" if flag else "")

    if node == "human_review":
        d = w.get("human_decision") or {}
        return f"Human decision: {d.get('decision')}" + (f" — {d['note']}" if d.get("note") else ""), [], ""
    if node == "finalize":
        return "Review saved to reviews_log.jsonl.", [], ""
    return "Done.", [], ""


def _progress(job: dict) -> int:
    if job.get("kind") == "train":
        t = job["training"]
        return min(95, 5 + int(90 * t["trials_done"] / max(t.get("max_trials") or 1, 1)))
    steps = job["steps"]
    total = len(TESTER_STEPS) + len(REVIEWER_STEPS) + 1 + (1 if job["settings"]["ask_human"] else 0)
    done = sum(s == "done" for group in steps.values() for s in group.values())
    return min(99, int(100 * done / total))


# The two nodes where an LLM reasons. Their full context is captured from the
# node's own input at start (the same functions the agents use to build it),
# so it can be inspected while the LLM is still working.
REASONING_STEPS = {("tester", "reason"), ("reviewer", "review")}


def _capture_context(job_id: str, agent: str, name: str, ev: dict, started: bool) -> None:
    from predict_agent.graph import REASON_SYSTEM, build_reason_payload
    from review_agent.graph import REVIEW_SYSTEM, build_review_payload

    key = f"{agent}.{name}"
    with JOBS_LOCK:
        job = JOBS[job_id]
        settings = job["settings"]
        ctx = dict(job["contexts"].get(key) or {})
    if started:
        model = (settings["tester_model"] if agent == "tester"
                 else settings["reviewer_model"] if settings["reviewer_model"] is not None
                 else settings["tester_model"])
        try:
            payload = (build_reason_payload if agent == "tester" else build_review_payload)(ev["input"])
        except Exception as exc:                      # never break the run for the viewer
            payload = {"error": f"context could not be rebuilt: {exc}"}
        ctx = {"step": key, "agent": ACTORS[agent], "node": name, "model": model or "",
               "status": "running", "started": time.time(), "finished": None,
               "system": REASON_SYSTEM if agent == "tester" else REVIEW_SYSTEM,
               "payload": payload, "output": None, "error": ""}
    else:
        w = _writes(ev)
        ctx.update(status="error" if ev.get("error") else "done", finished=time.time(),
                   error=ev.get("error") or w.get("llm_error") or "")
        if agent == "tester":
            ctx["output"] = {"llm_raw": w.get("llm_output") or None,
                             "final": w.get("final", []),
                             "rejected_overrides": w.get("rejected_overrides", []),
                             "reasoner": w.get("reasoner")}
        else:
            ctx["output"] = {"llm_raw": w.get("llm_output") or None,
                             "review": w.get("review"), "reasoner": w.get("reviewer")}
    with JOBS_LOCK:
        JOBS[job_id]["contexts"][key] = ctx


def on_task_event(job_id: str, ns: tuple, ev: dict) -> None:
    agent = ns[0].split(":", 1)[0] if ns else "orchestrator"
    name = ev["name"]
    started = "result" not in ev and "error" not in ev
    with JOBS_LOCK:
        job = JOBS[job_id]
        graph, steps = dict(job["graph"]), {k: dict(v) for k, v in job["steps"].items()}

    if agent == "orchestrator" and name in ("route", "clarify", "answer", "trainer"):
        _orchestrator_train_event(job_id, name, ev, started)
        return
    if agent == "trainer":
        _trainer_event(job_id, name, ev, started)
        return

    if agent == "orchestrator":
        target = {"tester": "tester", "reviewer": "reviewer",
                  "human_review": "human", "finalize": "orchestrator"}.get(name)
        if target is None:
            return
        if started:
            graph[target] = "active" if target != "human" else "waiting"
            if target == "orchestrator":
                steps["orchestrator"]["finalize"] = "active"
            msg = {"tester": "Handing the images to the Testing agent.",
                   "reviewer": "Handing the Testing agent's result to the Reviewer agent.",
                   "human_review": "The reviewer flagged issues: asking for a human decision.",
                   "finalize": "Closing the process and saving the review."}[name]
            activity(job_id, "Orchestrator", name, msg)
            update_job(job_id, graph=graph, steps=steps, stage=STEP_LABELS.get(name, msg))
            return
        if ev.get("interrupts"):
            return                                   # paused: handled after the stream ends
        failed = bool(ev.get("error")) or bool(_writes(ev).get("error"))
        if target == "orchestrator":
            steps["orchestrator"]["finalize"] = "error" if failed else "done"
        else:
            graph[target] = "error" if failed else "done"
        if target == "human":
            steps["human"]["decision"] = "done"
        if name in ("human_review", "finalize"):
            msg, details, level = _summarize(agent, name, _writes(ev), job)
            activity(job_id, ACTORS[target], name, msg, details, level)
        with JOBS_LOCK:
            job["graph"], job["steps"] = graph, steps
            job["percent"] = _progress(job)
        return

    group = steps.get(agent)
    if group is None or name not in group:
        return
    if (agent, name) in REASONING_STEPS:
        _capture_context(job_id, agent, name, ev, started)
    if started:
        group[name] = "active"
        graph[agent] = "active"
        update_job(job_id, graph=graph, steps=steps, stage=f"{ACTORS[agent]}: {STEP_LABELS[name]}")
        return
    w = _writes(ev)
    failed = bool(ev.get("error")) or bool(w.get("error"))
    group[name] = "error" if failed else "done"
    msg, details, level = _summarize(agent, name, w, job)
    if ev.get("error"):
        msg, level = f"Error: {ev['error']}", "error"
    activity(job_id, ACTORS[agent], name, msg, details, level)
    with JOBS_LOCK:
        job["graph"], job["steps"] = graph, steps
        job["percent"] = _progress(job)


def _orchestrator_train_event(job_id: str, name: str, ev: dict, started: bool) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        graph, steps = dict(job["graph"]), {k: dict(v) for k, v in job["steps"].items()}
    w = {} if started else _writes(ev)
    if name == "route":
        steps["orchestrator"]["route"] = "active" if started else "done"
        if not started:
            r = w.get("route", {})
            how = {"caller": "mode chosen by you", "images": "there are images",
                   "llm": "interpreted by the LLM", "no_router": "no LLM to interpret it",
                   "router_error": "router error"}.get(r.get("by"), r.get("by"))
            intent = {"predict": "prediction", "train": "training", "unclear": "unclear"}.get(r.get("intent"), r.get("intent"))
            details = [f"reason: {r.get('reason', '')}"]
            if r.get("extracted"):
                details.append(f"constraints extracted from the request: {r['extracted']}")
            activity(job_id, "Orchestrator", "route", f"Request routed: {intent} ({how}).", details)
    elif name == "clarify":
        if started:
            graph["human"] = "waiting"
            activity(job_id, "Orchestrator", "clarify", "It is not clear what you want to do: asking you.")
    elif name == "answer" and not started:
        activity(job_id, "Orchestrator", "answer", w.get("report", ""))
    elif name == "trainer":
        if not started and ev.get("interrupts"):
            return                     # the subgraph paused on a human approval: not finished
        graph["trainer"] = "active" if started else ("error" if ev.get("error") else "done")
        if started and not job.get("_trainer_announced"):
            # LangGraph re-runs the parent node on every resume: announce the handoff once
            job["_trainer_announced"] = True
            activity(job_id, "Orchestrator", "trainer", "Handing the request to the Training agent.")
    with JOBS_LOCK:
        job["graph"], job["steps"] = graph, steps


def _trainer_event(job_id: str, name: str, ev: dict, started: bool) -> None:
    step = "validate" if name == "give_up" else name
    with JOBS_LOCK:
        job = JOBS[job_id]
        graph, steps = dict(job["graph"]), {k: dict(v) for k, v in job["steps"].items()}
    group = steps["trainer"]
    if step not in group:
        return
    if started:
        group[step] = "active"
        graph["trainer"] = "active"
        update_job(job_id, graph=graph, steps=steps, stage=f"Training agent: {STEP_LABELS.get(step, step)}")
        return
    if ev.get("interrupts"):
        return                                         # paused on a human approval
    w = _writes(ev)
    group[step] = "error" if ev.get("error") else "done"
    if step.startswith("approve_") and graph.get("human") in ("waiting", "active"):
        graph["human"] = "done"
        steps["human"]["decision"] = "done"
    for e in w.get("events", []):
        activity(job_id, "Training agent", e["node"], e["message"], e.get("details"), e.get("level", ""))
    if ev.get("error"):
        activity(job_id, "Training agent", step, f"Error: {ev['error']}", level="error")
    with JOBS_LOCK:
        t = job["training"]
        if step == "plan" and w.get("plan"):
            t["plan"], t["max_trials"], t["campaign_id"] = w["plan"], w["plan"]["max_trials"], w.get("campaign_id")
        if step == "approve_plan" and w.get("plan"):
            t["plan"], t["max_trials"] = w["plan"], w["plan"]["max_trials"]
        if step == "analyse" and w.get("trials"):
            t["trials_done"] += len(w["trials"])
            for tr in w["trials"]:
                t["trials"].append(_compact_trial(tr))
        job["graph"], job["steps"] = graph, steps
        job["percent"] = _progress(job)


def _compact_trial(tr: dict) -> dict:
    s, p = tr["summary"], tr["proposal"]
    return {"run": s["run"], "action": p["action"], "arch": p["arch"], "parent_run": p.get("parent_run"),
            "lr": p["hparams"]["lr"], "weight_decay": p["hparams"]["weight_decay"],
            "class_weight": p["hparams"]["class_weight"], "augmentation": p["augmentation"]["preset"],
            "status": s["status"], "best_score": s.get("best_score"), "best_epoch": s.get("best_epoch"),
            "epochs_completed": s.get("epochs_completed"), "stop_reason": s.get("stop_reason") or s.get("error"),
            "per_class_recall": s.get("per_class_recall"), "flags": tr["facts"].get("flags", []),
            "verdict": (tr.get("diagnosis") or {}).get("verdict", ""), "rationale": p.get("rationale", ""),
            "augmentation_rationale": p.get("augmentation_rationale", "")}


def on_custom_event(job_id: str, data: dict) -> None:
    """Per-epoch rows streamed by the training agent's train node."""
    if not isinstance(data, dict) or data.get("type") != "epoch":
        return
    keep = {k: data.get(k) for k in ("epoch", "train_loss", "val_loss", "val_balanced_acc",
                                      "val_macro_f1", "val_acc", "lr", "epoch_seconds")}
    with JOBS_LOCK:
        job = JOBS[job_id]
        t = job["training"]
        t["runs"].setdefault(data["run"], []).append(keep)
        t["current"] = data["run"]
        job["stage"] = (f"Training agent: {data['run']} · epoch {data.get('epoch')} · "
                        f"val bal.acc {float(data.get('val_balanced_acc') or 0):.3f}")


def _train_payload(values: dict) -> dict:
    tr = values.get("train_result") or {}
    return {"kind": "train", "report": values.get("report", ""), "route": values.get("route"),
            "campaign_id": tr.get("campaign_id"), "plan": tr.get("plan"),
            "stop_reason": tr.get("stop_reason"),
            "trials": [_compact_trial(t) for t in tr.get("trials", [])],
            "candidate": {k: (tr.get("candidate") or {}).get(k) for k in ("run", "arch", "best_score", "dir")},
            "evaluation": tr.get("evaluation") or {}, "promotion": tr.get("promotion") or {}}


def _result_payload(values: dict) -> dict:
    test, rv = values.get("test_result") or {}, values.get("review_result") or {}
    review = rv.get("review") or {}
    images = []
    for i, f in enumerate(test.get("final", [])):
        v = test["votes"][i]
        r = (review.get("images") or [{}] * (i + 1))[i]
        found = (rv.get("checks") or [[]] * (i + 1))[i]
        order = {"critical": 0, "warning": 1, "info": 2}
        issues = sorted(found + r.get("issues", []), key=lambda x: order[x["severity"]])
        d = (rv.get("decisive") or [{}] * (i + 1))[i]
        images.append({
            "index": i, "file": f["file"], "final_class": f["final_class"],
            "confidence_level": f["confidence_level"], "overridden": f["overridden"],
            "vote_class": f["vote_class"], "rationale": f["rationale"],
            "verdict": r.get("verdict", ""), "flagged": any(x["severity"] != "info" for x in issues),
            "user_summary": r.get("user_summary", ""),
            "decisive_model_explanation": r.get("decisive_model_explanation", ""),
            "decisive": {k: d.get(k) for k in ("model", "share_of_final_class_score",
                                               "strongest_model", "strongest_model_agrees")},
            "soft_vote": dict(list(v["soft_vote"].items())[:5]),
            "issues": issues,
            "sources": f.get("sources", []),
            "past": f.get("past_predictions", [])[:1],
        })
    return {
        "report": values.get("report", ""),
        "error": values.get("error", ""),
        "needs_human_review": values.get("needs_human_review", False),
        "human_decision": values.get("human_decision"),
        "process_summary": review.get("process_summary", ""),
        # small local models sometimes put a bare severity word here
        "overall": "" if str(review.get("overall", "")).strip().lower() in {"", "info", "warning", "critical"}
                   else review["overall"],
        "tester": test.get("reasoner"), "reviewer": rv.get("reviewer"),
        "execution_id": test.get("execution_id"),
        "images": images,
    }


def run_stream(job_id: str, payload) -> None:
    """Run (or resume) the orchestrator for one job, translating its task stream."""
    with JOBS_LOCK:
        job = JOBS[job_id]
        settings, config, kind = job["settings"], job["config"], job.get("kind", "predict")
    as_payload = _train_payload if kind == "train" else _result_payload
    try:
        with (TRAIN_LOCK if kind == "train" else RUN_LOCK):
            app = get_app(settings, train=kind == "train")
            for ns, mode, ev in app.stream(payload, config, stream_mode=["tasks", "custom"], subgraphs=True):
                if mode == "custom":
                    on_custom_event(job_id, ev)
                else:
                    on_task_event(job_id, ns, ev)
            snap = app.get_state(config)
        pending = [i for t in snap.tasks for i in (t.interrupts or [])]
        if pending:
            with JOBS_LOCK:
                graph = dict(JOBS[job_id]["graph"])
                graph["human"] = "waiting"
                steps = {k: dict(v) for k, v in JOBS[job_id]["steps"].items()}
                steps["human"]["decision"] = "active"
            update_job(job_id, status="waiting_human", interrupt=pending[0].value,
                       graph=graph, steps=steps,
                       result=as_payload(snap.values),
                       stage="Waiting for your decision")
            return
        values = snap.values
        with JOBS_LOCK:
            graph = dict(JOBS[job_id]["graph"])
        failed = bool(values.get("error"))
        graph["orchestrator"] = "error" if failed else "done"
        if graph.get("human") == "idle":
            graph["human"] = "skipped"
        if kind == "train" and graph.get("trainer") == "active":
            graph["trainer"] = "done"
        activity(job_id, "Orchestrator", "done",
                 "Process stopped: " + values["error"] if failed
                 else "Result verified and returned.", level="error" if failed else "")
        update_job(job_id, status="error" if failed else "complete", percent=100,
                   stage="Process stopped" if failed else "Result ready",
                   graph=graph, result=as_payload(values), interrupt=None,
                   error=values.get("error", ""))
        HEALTH["time"] = 0.0                       # logs changed
    except Exception as exc:
        traceback.print_exc()
        with JOBS_LOCK:
            job = JOBS[job_id]
            graph = dict(job["graph"])
            steps = {k: {s: ("error" if st == "active" else st) for s, st in v.items()}
                     for k, v in job["steps"].items()}
        for k, st in graph.items():
            if st in ("active", "waiting"):
                graph[k] = "error"
        activity(job_id, "Orchestrator", "error", f"The process stopped: {exc}", level="error")
        update_job(job_id, status="error", stage="Process stopped", error=str(exc),
                   graph=graph, steps=steps)


def _save_uploads(uploads: list[dict], folder: Path) -> list[Path]:
    paths = []
    for n, up in enumerate(uploads):
        encoded = str(up.get("data") or "")
        if "," not in encoded:
            raise ValueError("invalid image upload")
        raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("each image must be at most 10 MB")
        name = Path(str(up.get("name") or f"image{n}.png")).name
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError("use PNG, JPG, BMP or TIFF images")
        stem = re.sub(r"[^\w.-]+", "_", Path(name).stem)[:60] or f"image{n}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{n + 1:02d}_{stem}{suffix}"
        path.write_bytes(raw)
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            path.unlink(missing_ok=True)
            raise ValueError(f"{name} is not a readable image")
        paths.append(path)
    return paths


def _train_constraints(raw: dict | None) -> dict:
    from train_agent.space import AUTONOMY, SELECT_ON
    raw, out = raw or {}, {}
    arch = str(raw.get("arch") or "").strip()
    if arch and arch != "auto":
        if arch not in _archs():
            raise ValueError(f"unknown architecture: {arch}")
        out["arch"] = arch
    for key, lo, hi, cast in (("max_trials", 1, 20, int), ("max_minutes", 1, 1440, float),
                              ("max_epochs_per_run", 1, 200, int), ("patience", 1, 50, int)):
        if raw.get(key) not in (None, ""):
            v = cast(raw[key])
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be between {lo} and {hi}")
            out[key] = v
    if raw.get("target_score") not in (None, ""):
        v = float(raw["target_score"])
        if not 0 < v < 1:
            raise ValueError("target_score must be between 0 and 1")
        out["target_score"] = v
    if raw.get("autonomy") in AUTONOMY:
        out["autonomy"] = raw["autonomy"]
    if raw.get("select_on") in SELECT_ON:
        out["select_on"] = raw["select_on"]
    return out                                        # auto_approve_plan is never set from the web


def start_train_job(payload: dict, settings: dict) -> str:
    """A training campaign (mode 'train'), or free text the orchestrator must route (mode '')."""
    job_id = uuid.uuid4().hex[:12]
    mode = "train" if payload.get("mode") == "train" else ""
    request = str(payload.get("request") or "")[:2000]
    constraints = _train_constraints(payload.get("train"))
    job = {
        "id": job_id, "kind": "train", "status": "running", "percent": 2, "stage": "Preparing the request",
        "settings": settings, "config": {"configurable": {"thread_id": job_id}}, "images": [],
        "graph": {"orchestrator": "active", "trainer": "idle", "human": "idle",
                  "tester": "skipped", "reviewer": "skipped"},
        "steps": {"trainer": {s: "idle" for s in TRAINER_STEPS}, "human": {"decision": "idle"},
                  "orchestrator": {"route": "idle", "finalize": "skipped"},
                  "tester": {}, "reviewer": {}},
        "training": {"plan": None, "max_trials": constraints.get("max_trials"), "trials_done": 0,
                     "trials": [], "runs": {}, "current": None, "campaign_id": None},
        "activities": [], "result": None, "error": "", "interrupt": None, "contexts": {},
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    tm = settings["trainer_model"] if settings["trainer_model"] is not None else settings["tester_model"]
    activity(job_id, "Orchestrator", "start",
             ("Training request accepted." if mode == "train" else "Text request: routing it.")
             + f" Training agent: {tm or 'deterministic policy (no LLM)'}.",
             [f"constraints: {constraints or 'none (default values, editable in the plan)'}"])
    if TRAIN_LOCK.locked():
        activity(job_id, "Orchestrator", "queue", "Another campaign is running: this one starts as soon as it ends.")
    threading.Thread(target=run_stream, daemon=True,
                     args=(job_id, {"request": request, "mode": mode, "constraints": constraints})).start()
    return job_id


def start_job(payload: dict) -> str:
    settings = normalize_settings(payload.get("settings"))
    has_images = bool(payload.get("samples") or payload.get("uploads"))
    if payload.get("mode") == "train" or (not has_images and str(payload.get("request") or "").strip()):
        return start_train_job(payload, settings)
    job_id = uuid.uuid4().hex[:12]
    samples = [str(s) for s in payload.get("samples") or []]
    available = set(health()["samples"])
    if samples == ["*"]:
        samples = sorted(available)
    unknown = [s for s in samples if s not in available]
    if unknown:
        raise ValueError(f"unknown samples: {unknown}")
    paths = [SAMPLES_DIR / s for s in samples]
    paths += _save_uploads(list(payload.get("uploads") or []), UPLOAD_ROOT / job_id)
    if not paths:
        raise ValueError("attach at least one image or pick a sample")
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per request")

    job = {
        "id": job_id, "status": "running", "percent": 1, "stage": "Preparing the request",
        "settings": settings, "config": {"configurable": {"thread_id": job_id}},
        "images": [{"file": p.name, "path": str(p)} for p in paths],
        "graph": {"orchestrator": "active", "tester": "idle", "reviewer": "idle",
                  "human": "idle" if settings["ask_human"] else "skipped"},
        "kind": "predict",
        "steps": {"tester": {s: "idle" for s in TESTER_STEPS},
                  "reviewer": {s: "idle" for s in REVIEWER_STEPS},
                  "human": {"decision": "idle" if settings["ask_human"] else "skipped"},
                  "orchestrator": {"route": "idle", "finalize": "idle"}},
        "activities": [], "result": None, "error": "", "interrupt": None, "contexts": {},
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    who = settings["tester_model"] or "vote only"
    rv = (settings["reviewer_model"] if settings["reviewer_model"] is not None
          else settings["tester_model"]) or "automatic checks only"
    activity(job_id, "Orchestrator", "start",
             f"Request accepted: {_n(len(paths), 'image', 'images')}. Testing agent: {who}; reviewer: {rv}.")
    if RUN_LOCK.locked():
        activity(job_id, "Orchestrator", "queue", "Another process is running: this one starts as soon as it ends.")
    request = str(payload.get("request") or "")[:2000]
    threading.Thread(target=run_stream, daemon=True,
                     args=(job_id, {"image_paths": [str(p) for p in paths], "request": request})).start()
    return job_id


REVIEW_DECISIONS = {"accepted": "Predictions accepted.", "rejected": "Predictions rejected.",
                    "accepted_with_note": "Predictions accepted with a note."}


def resume_job(payload: dict) -> None:
    """Answer whatever the graph is waiting on. Each interrupt declares its own
    `options`; the reviewer's (which predates them) keeps its three decisions."""
    job_id = str(payload.get("job") or "")
    decision = str(payload.get("decision") or "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job["status"] != "waiting_human":
            raise ValueError("no process is waiting for a decision")
        pending = job.get("interrupt") or {}
        options = {o["value"]: o["label"] for o in pending.get("options", [])} or REVIEW_DECISIONS
        if decision not in options:
            raise ValueError("invalid decision")
        job["status"] = "running"
        job["interrupt"] = None
        job["graph"]["human"] = "active"
    label = REVIEW_DECISIONS.get(decision) if pending.get("options") is None else f"Decision: {options[decision]}."
    note = str(payload.get("note") or "")[:1000]
    resume = {"decision": decision, "note": note}
    edits = payload.get("edits")
    if pending.get("kind") == "approve_plan" and isinstance(edits, dict):
        resume["edits"] = {k: v for k, v in edits.items() if k in set(pending.get("editable", []))}
    activity(job_id, "Human", pending.get("kind") or "decision", label,
             ([f"note: {note}"] if note else []) + ([f"edits: {resume['edits']}"] if resume.get("edits") else []))
    threading.Thread(target=run_stream, args=(job_id, Command(resume=resume)), daemon=True).start()


def public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("config", "_files", "contexts", "_trainer_announced")} | {
        "context_steps": sorted(job.get("contexts", {}))}


def knowledge_base():
    from predict_agent.memory import KnowledgeBase
    with KB_LOCK:
        if "kb" not in KB:
            KB["kb"] = KnowledgeBase(CORPUS)
        return KB["kb"]


def kb_overview() -> dict:
    from predict_agent.memory import CLASS_QUERIES
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    sources = [{"id": s["id"], "title": s.get("title"), "citation": s.get("citation"),
                "url": s.get("url"), "chunks": s.get("chunks", [])}
               for s in corpus.get("sources", [])]
    lengths = [len(c) for s in sources for c in s["chunks"]]
    return {
        "created_at": corpus.get("created_at"), "failures": corpus.get("failures", []),
        "sources": sources,
        "stats": {"sources": len(sources), "chunks": len(lengths),
                  "avg_chars": round(sum(lengths) / max(len(lengths), 1)),
                  "total_chars": sum(lengths)},
        "classes": [{"id": i, "name": n, "query": f"{n} {CLASS_QUERIES[i]} dermoscopy"}
                    for i, n in enumerate(CLASS_NAMES)],
        "method": "TF-IDF on unigrams and bigrams (sublinear tf, English stop words), with the "
                  "source title prepended to each passage; query expanded with the class's "
                  "dermoscopic terms; cosine similarity.",
    }


def kb_search(query: str, k: int) -> dict:
    query = query.strip()[:500]
    if not query:
        raise ValueError("empty query")
    kb = knowledge_base()
    hits = kb.search(query, k=max(1, min(k, 20)))
    full = {c["chunk_id"]: c["text"] for c in kb.chunks}
    return {"query": query, "results": [{**h, "text": full[h["chunk_id"]]} for h in hits]}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload) -> None:
        self.send_bytes(status, "application/json; charset=utf-8",
                        json.dumps(payload, ensure_ascii=False, default=str).encode())

    def read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES * MAX_IMAGES * 2:
            raise ValueError("empty or too large request")
        return json.loads(self.rfile.read(length))

    def _image(self, path: Path) -> None:
        ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}.get(
            path.suffix.lower(), "application/octet-stream")
        self.send_bytes(200, ctype, path.read_bytes())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in PAGES:
            self.send_bytes(200, "text/html; charset=utf-8", PAGES[parsed.path].read_bytes())
        elif parsed.path == "/api/context":
            with JOBS_LOCK:
                job = JOBS.get(query.get("job", [""])[0])
                ctx = job["contexts"].get(query.get("step", [""])[0]) if job else None
                meta = ({"job": job["id"], "job_status": job["status"],
                         "images": [i["file"] for i in job["images"]],
                         "available": sorted(job["contexts"])} if job else None)
                data = json.loads(json.dumps({**meta, "context": ctx}, default=str)) if job else None
            self.send_json(200 if data else 404, data or {"error": "job not found"})
        elif parsed.path == "/api/kb":
            self.send_json(200, kb_overview())
        elif parsed.path == "/api/kb/search":
            try:
                self.send_json(200, kb_search(query.get("q", [""])[0],
                                              int(query.get("k", ["8"])[0] or 8)))
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
        elif parsed.path == "/favicon.ico":
            self.send_bytes(204, "image/x-icon", b"")
        elif parsed.path == "/api/health":
            self.send_json(200, health(force=query.get("refresh", ["0"])[0] == "1"))
        elif parsed.path == "/api/status":
            with JOBS_LOCK:
                job = JOBS.get(query.get("id", [""])[0])
                data = json.loads(json.dumps(public_job(job), default=str)) if job else None
            self.send_json(200 if data else 404, data or {"error": "job not found"})
        elif parsed.path == "/api/image":
            with JOBS_LOCK:
                job = JOBS.get(query.get("job", [""])[0])
            try:
                path = Path(job["images"][int(query.get("i", ["-1"])[0])]["path"]) if job else None
            except (ValueError, IndexError):
                path = None
            if path and path.is_file():
                self._image(path)
            else:
                self.send_json(404, {"error": "image not found"})
        elif parsed.path == "/api/sample":
            name = Path(query.get("name", [""])[0]).name
            path = SAMPLES_DIR / name
            if name in health()["samples"] and path.is_file():
                self._image(path)
            else:
                self.send_json(404, {"error": "sample not found"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self.read_payload()
            if self.path == "/api/run":
                self.send_json(202, {"job": start_job(payload)})
            elif self.path == "/api/resume":
                resume_job(payload)
                self.send_json(202, {"ok": True})
            else:
                self.send_json(404, {"error": "not found"})
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, OSError) as exc:
            self.send_json(400, {"error": str(exc)})


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    health(force=True)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"DermaAgent platform: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
