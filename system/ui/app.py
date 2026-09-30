"""
DermaAgent platform: the orchestrator (testing agent -> reviewer -> human, or the
training agent) with a live view of which agent is working and what it is doing.

    python run.py                    # http://127.0.0.1:8000, opens the browser
    python run.py --port 8080

Standard library HTTP server, a few static pages, jobs in background threads;
the browser polls `/api/status`. Nothing in the agents was written for this UI:
the live graph comes from LangGraph's own task stream (`stream_mode="tasks",
subgraphs=True`), which reports the start and the result of every node - the
orchestrator's and those of the agents running inside it, each tagged with its
namespace. Human decisions are the graphs' `interrupt()`s; the page answers and
the job resumes with `Command(resume=...)`.

Only images uploaded through the page or the files in `input/samples/` can be
analysed: no path from the browser ever reaches the filesystem. The server binds
to 127.0.0.1 and refuses cross-origin writes, because it can store API keys.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from PIL import Image

import llm as providers
import paths
from tester.models import CLASS_NAMES, IMAGE_SUFFIXES, PROJECT_ROOT, ModelZoo

HERE = Path(__file__).parent
PAGES = {"/": HERE / "page.html", "/context": HERE / "context.html", "/kb": HERE / "kb.html",
         "/about": HERE / "about.html"}
SAMPLES_DIR = paths.SAMPLES
UPLOAD_ROOT = paths.UPLOADS
PREDICTIONS_LOG = paths.PREDICTIONS_LOG
REVIEWS_LOG = paths.REVIEWS_LOG
CORPUS = paths.CLINICAL_KB

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 16
MODEL_RE = re.compile(r"^(?:[a-z0-9][a-z0-9_-]{0,31}:[\w.\-:/@]{1,160}|claude-[\w.\-]{1,80})$")

TESTER_STEPS = ["load_inputs", "recall_memory", "run_models", "vote",
                "retrieve_knowledge", "reason", "write_log"]
REVIEWER_STEPS = ["gather", "review", "render"]
TRAINER_STEPS = ["plan", "approve_plan", "propose", "validate", "review_proposal", "approve_proposal",
                 "train", "analyse", "decide", "evaluate", "approve_promotion", "promote", "finish"]
STEP_LABELS = {
    "load_inputs": "Loading the images", "recall_memory": "Checking past predictions",
    "run_models": "Running the models", "vote": "Combining the model votes",
    "retrieve_knowledge": "Searching the clinical knowledge base", "reason": "Writing the rationale",
    "write_log": "Saving the prediction", "gather": "Running the automatic checks",
    "review": "Checking the rationale against the evidence", "render": "Writing the report",
    "finalize": "Saving the review", "human_review": "Waiting for your decision",
    "route": "Reading the request", "clarify": "Waiting for you to choose an action",
    "answer": "Answering", "explain": "Writing the answer", "trainer": "Starting the Training agent",
    "plan": "Preparing the plan", "approve_plan": "Waiting for plan approval",
    "propose": "Proposing the next run", "validate": "Validating the proposal",
    "review_proposal": "Checking the proposal and the autonomy level",
    "approve_proposal": "Waiting for run approval", "train": "Training",
    "analyse": "Analysing the training curves", "decide": "Deciding whether to continue",
    "evaluate": "Comparing the ensemble with and without the candidate",
    "approve_promotion": "Waiting for the promotion decision", "promote": "Adding the model to the ensemble",
    "finish": "Writing the campaign report",
    "inspector": "Starting the Models agent", "collect_facts": "Collecting the model facts (validation only)",
    "check_claims": "Checking the answer against the facts",
}
ACTORS = {"orchestrator": "Orchestrator", "tester": "Testing agent",
          "reviewer": "Reviewer agent", "trainer": "Training agent", "human": "You",
          "inspector": "Models agent"}
MODELS_STEPS = ["collect_facts", "explain", "check_claims"]

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()          # one prediction at a time: models and LLMs are shared
TRAIN_LOCK = threading.Lock()        # one training campaign at a time; predictions do not wait for it
APPS: dict[str, Any] = {}
APPS_LOCK = threading.Lock()
KB: dict[str, Any] = {}
KB_LOCK = threading.Lock()
HEALTH: dict[str, Any] = {"time": 0.0}


# ---------------------------------------------------------------------------
# Environment / health
# ---------------------------------------------------------------------------

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _archs() -> dict:
    from nets.zoo import ARCHITECTURES
    return ARCHITECTURES


def _llm_summary() -> dict:
    overview = providers.providers_overview()
    return {"options": providers.available_models(), "default": overview["resolved_default"],
            "providers": [{"id": p["id"], "label": p["label"], "ready": p["ready"]}
                          for p in overview["providers"]]}


def health(force: bool = False) -> dict:
    """Cached for 30 s: scanning the checkpoints loads them from disk."""
    if not force and time.time() - HEALTH["time"] < 30:
        return HEALTH["data"]
    try:
        zoo = ModelZoo()
        models = [{"name": c.name, "epoch": c.epoch, "val_balanced_acc": round(c.val_balanced_acc, 3),
                   "skill": round(c.skill, 3), "blind_classes": [CLASS_NAMES[i] for i in c.blind_classes],
                   "arch": c.config.get("arch") or "fpvit",
                   "folder": str(Path(c.path).parent.relative_to(PROJECT_ROOT))}
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
    data = {
        "models": models, "knowledge": kb, "llm": _llm_summary(),
        "logs": {"predictions": _count_lines(PREDICTIONS_LOG), "reviews": _count_lines(REVIEWS_LOG)},
        "archs": _archs(),
        "samples": sorted(p.name for p in SAMPLES_DIR.iterdir()
                          if p.suffix.lower() in IMAGE_SUFFIXES) if SAMPLES_DIR.is_dir() else [],
        "classes": CLASS_NAMES,
        "campaign_running": _campaign_active(),
    }
    HEALTH.update(time=time.time(), data=data)
    return data


def _invalidate() -> None:
    """Providers or keys changed: rebuild orchestrators on next use, refresh health."""
    with APPS_LOCK:
        APPS.clear()
    HEALTH["time"] = 0.0


# ---------------------------------------------------------------------------
# Orchestrators, cached per settings
# ---------------------------------------------------------------------------

def normalize_settings(raw: dict | None) -> dict:
    """Model fields: "auto" -> the default model now, "" / "none" -> no LLM,
    "same" -> the Testing agent's model (reviewer and trainer only)."""
    raw = raw or {}

    def model_field(value, allow_same: bool):
        value = str(value or "").strip()
        if value in ("", "none"):
            return ""
        if allow_same and value == "same":
            return None
        if value == "auto":
            return providers.default_model() or ""
        if not MODEL_RE.match(value):
            raise ValueError(f"'{value}' is not a model address. Use provider:model.")
        return value

    min_acc = float(raw.get("minBalancedAcc") or 0.0)
    if not 0.0 <= min_acc < 1.0:
        raise ValueError("The minimum balanced accuracy must be between 0 and 1.")
    return {
        "tester_model": model_field(raw.get("testerModel", "auto"), allow_same=False),
        "reviewer_model": model_field(raw.get("reviewerModel", "same"), allow_same=True),
        "trainer_model": model_field(raw.get("trainerModel", "same"), allow_same=True),
        "explain": bool(raw.get("explain", True)),
        "ask_human": bool(raw.get("askHuman", True)),
        "min_balanced_acc": min_acc,
    }


def get_app(settings: dict, lane: str = "predict"):
    """One compiled orchestrator per configuration and lane, kept for the process life.

    lane "predict": tester + reviewer. "train": adds the training branch and
    intent routing. "chat": routing and answers without the training branch, used
    while a campaign is already running.
    """
    from orchestrator import build_orchestrator

    key = json.dumps(settings, sort_keys=True) + "|" + lane
    with APPS_LOCK:
        if key not in APPS:
            tm = settings["trainer_model"]
            extra = {}
            if lane == "train":
                extra = {"with_trainer": True, "trainer_model": tm if tm is not None else settings["tester_model"]}
            elif lane == "chat":
                extra = {"with_router": True,
                         "trainer_model": tm if tm is not None else settings["tester_model"],
                         "trainer_unavailable": "A training campaign is already running. Start a new one when it "
                                                "finishes, or stop it at its next approval step."}
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


def _campaign_active() -> bool:
    with JOBS_LOCK:
        return any(j.get("campaign") and j["status"] in ("running", "waiting_human") for j in JOBS.values())


def update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def activity(job_id: str, actor: str, node: str, message: str, details: list[str] | None = None,
             level: str = "") -> None:
    with JOBS_LOCK:
        items = JOBS[job_id]["activities"]
        items.append({"actor": actor, "node": node, "message": message,
                      "details": details or [], "level": level, "time": _now()})
        del items[:-120]


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
        return f"Stopped: {w['error']}", [], "error"

    if agent == "tester":
        trace = [ln for ln in w.get("trace", [])[1:] if not ln.strip().startswith("(")]
        if node == "load_inputs":
            job["_files"] = [i["file"] for i in w.get("images", [])]
            return f"Loaded {_n(len(job['_files']), 'image', 'images')}.", trace, ""
        if node == "recall_memory":
            seen = sum(1 for v in (w.get("past") or {}).values() if v)
            stats = w.get("log_stats") or {}
            return (f"Seen before: {seen} of {_n(len(files), 'image', 'images')}. "
                    f"The prediction log holds {_n(stats.get('images', 0), 'entry', 'entries')}.", trace, "")
        if node == "run_models":
            names = [m["name"] for m in w.get("models", [])]
            return f"Ran {_n(len(names), 'model', 'models')}: {', '.join(names)}.", trace, ""
        if node == "vote":
            parts = [f"{f}: {v['vote_class']} (p={v['vote_probability']:.2f}, {v['confidence_level']} confidence)"
                     for f, v in zip(files, w.get("votes", []))]
            return "Ensemble vote. " + "; ".join(parts) + ".", trace, ""
        if node == "retrieve_knowledge":
            ids = {c["chunk_id"] for ev in w.get("evidence", []) for c in ev}
            return f"Retrieved {_n(len(ids), 'passage', 'passages')} from the clinical knowledge base.", trace, ""
        if node == "reason":
            who = w.get("reasoner", "")
            parts = [f"{f['file']}: {f['final_class']}" + (" (changed from the vote)" if f["overridden"] else "")
                     for f in w.get("final", [])]
            level = "warn" if w.get("llm_error") or w.get("rejected_overrides") else ""
            head = ("No language model: the ensemble vote is the final answer" if who == "deterministic"
                    else f"Rationale written by {who}")
            if w.get("llm_error"):
                head = f"The language model failed ({w['llm_error'][:120]}), so the vote was kept"
            return f"{head}. " + "; ".join(parts) + ".", trace, level
        if node == "write_log":
            return f"Saved to the prediction log (execution {w.get('execution_id')}).", trace, ""

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
                details.append("  independent passages: " + ", ".join(c["chunk_id"] for c in kb))
                details += [f"  [{c['severity']}] {c['description']}" for c in found]
            return (f"Automatic checks: {_n(crit, 'critical finding', 'critical findings')}, "
                    f"{_n(warn, 'warning', 'warnings')}. Decisive model: {', '.join(sorted(dec))}.", details,
                    "error" if crit else "warn" if warn else "")
        if node == "review":
            rv = w.get("review", {})
            verdicts = [f"{r['file']}: {'consistent' if r['verdict'] == 'consistent' else 'needs review'}"
                        for r in rv.get("images", [])]
            details = [f"{r['file']}: [{i['severity']}] {i['description']}"
                       + (f" ({i['evidence']})" if i.get("evidence") else "")
                       for r in rv.get("images", []) for i in r.get("issues", [])]
            if w.get("llm_error"):
                details.append(f"Language model error: {w['llm_error']}")
            who = w.get("reviewer", "")
            head = "Review from the automatic checks only" if who == "deterministic" else f"Review by {who}"
            return f"{head}. " + "; ".join(verdicts) + ".", details, "warn" if details else ""
        if node == "render":
            flag = w.get("needs_human_review")
            return ("Report ready. " + ("Some results are flagged for your review." if flag
                                        else "No inconsistency found."), [], "warn" if flag else "")

    if agent == "inspector":
        if node == "collect_facts":
            facts = w.get("facts") or {}
            members = [m["name"] for m in facts.get("models", []) if m.get("role") == "ensemble member"]
            excluded = [m["name"] for m in facts.get("models", []) if m.get("role") == "excluded"]
            vs = facts.get("validation_set") or {}
            return (f"Collected the facts of {_n(len(members), 'ensemble member', 'ensemble members')} and "
                    f"{_n(len(excluded), 'excluded run', 'excluded runs')} on {vs.get('name', 'validation')} "
                    f"({vs.get('images', '?')} images).",
                    [f"ensemble: {', '.join(members)}", f"excluded: {', '.join(excluded) or 'none'}"], "")
        if node == "explain":
            who = w.get("answered_by", "")
            if w.get("llm_error"):
                return (f"The language model failed ({w['llm_error'][:120]}), so the answer is the summary "
                        "computed in code.", [], "warn")
            return ("Answer from the summary computed in code (no language model)." if who == "deterministic"
                    else f"Answer written by {who} from the facts."), [], ""
        if node == "check_claims":
            c = w.get("claims") or {}
            if c.get("verdict") == "rejected":
                return ("The answer cited figures or names that are not in the facts, so it was replaced by the "
                        "summary computed in code.", [f"not in the facts: {', '.join(c.get('unsupported', []))}"],
                        "warn")
            return ("Every figure and name in the answer is in the facts." if c.get("verdict") == "supported"
                    else "Answer computed in code: nothing to check."), [], ""

    if node == "human_review":
        d = w.get("human_decision") or {}
        label = {"accepted": "accepted", "rejected": "rejected",
                 "accepted_with_note": "accepted with a note"}.get(d.get("decision"), d.get("decision"))
        return f"Decision recorded: {label}." + (f" Note: {d['note']}" if d.get("note") else ""), [], ""
    if node == "finalize":
        return "Review saved to the review log.", [], ""
    return "Done.", [], ""


def _progress(job: dict) -> int:
    if job.get("view") == "models":
        done = sum(s == "done" for s in job["steps"].get("inspector", {}).values())
        return min(99, 10 + 30 * done)
    if job.get("view") == "train":
        t = job["training"]
        return min(95, 5 + int(90 * t["trials_done"] / max(t.get("max_trials") or 1, 1)))
    if job.get("view") != "predict":
        return 50
    steps = job["steps"]
    total = len(TESTER_STEPS) + len(REVIEWER_STEPS) + 1 + (1 if job["settings"]["ask_human"] else 0)
    done = sum(s == "done" for group in steps.values() for s in group.values())
    return min(99, int(100 * done / total))


# The nodes where an LLM reasons. Their full context is captured from the node's
# own input at start (the same functions the agents use to build it), so it can
# be inspected while the LLM is still working.
REASONING_STEPS = {("tester", "reason"), ("reviewer", "review"), ("inspector", "explain")}


def _capture_context(job_id: str, agent: str, name: str, ev: dict, started: bool) -> None:
    from informer.graph import EXPLAIN_SYSTEM, build_explain_payload
    from tester.graph import REASON_SYSTEM, build_reason_payload
    from reviewer.graph import REVIEW_SYSTEM, build_review_payload

    systems = {"tester": REASON_SYSTEM, "reviewer": REVIEW_SYSTEM, "inspector": EXPLAIN_SYSTEM}
    builders = {"tester": build_reason_payload, "reviewer": build_review_payload,
                "inspector": build_explain_payload}
    key = f"{agent}.{name}"
    with JOBS_LOCK:
        job = JOBS[job_id]
        settings = job["settings"]
        ctx = dict(job["contexts"].get(key) or {})
    if started:
        model = (settings["reviewer_model"] if agent == "reviewer" and settings["reviewer_model"] is not None
                 else settings["tester_model"])       # the Models agent answers with the tester's model
        try:
            payload = builders[agent](ev["input"])
        except Exception as exc:                      # never break the run for the viewer
            payload = {"error": f"context could not be rebuilt: {exc}"}
        ctx = {"step": key, "agent": ACTORS[agent], "node": name, "model": model or "",
               "status": "running", "started": time.time(), "finished": None,
               "system": systems[agent], "payload": payload, "output": None, "error": ""}
    else:
        w = _writes(ev)
        ctx.update(status="error" if ev.get("error") else "done", finished=time.time(),
                   error=ev.get("error") or w.get("llm_error") or "")
        if agent == "tester":
            ctx["output"] = {"llm_raw": w.get("llm_output") or None,
                             "final": w.get("final", []),
                             "rejected_overrides": w.get("rejected_overrides", []),
                             "reasoner": w.get("reasoner")}
        elif agent == "inspector":
            ctx["output"] = {"llm_raw": w.get("llm_output") or None, "answer": w.get("answer", ""),
                             "reasoner": w.get("answered_by"), "claims": None}
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

    if agent == "orchestrator" and name in ("route", "clarify", "answer", "trainer", "explain", "inspector"):
        _orchestrator_text_event(job_id, name, ev, started)
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
            msg = {"tester": "Sent the images to the Testing agent.",
                   "reviewer": "Sent the Testing agent's result to the Reviewer agent.",
                   "human_review": "The Reviewer flagged some results. Waiting for your decision.",
                   "finalize": "Saving the review."}[name]
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
    if (agent, name) == ("inspector", "check_claims"):       # the verdict belongs to the explain context
        with JOBS_LOCK:
            ctx = job["contexts"].get("inspector.explain")
            if ctx and ctx.get("output") is not None:
                ctx["output"]["claims"] = w.get("claims")
    msg, details, level = _summarize(agent, name, w, job)
    if ev.get("error"):
        msg, level = f"Stopped: {ev['error']}", "error"
    activity(job_id, ACTORS[agent], name, msg, details, level)
    with JOBS_LOCK:
        job["graph"], job["steps"] = graph, steps
        job["percent"] = _progress(job)


def _orchestrator_text_event(job_id: str, name: str, ev: dict, started: bool) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        graph, steps = dict(job["graph"]), {k: dict(v) for k, v in job["steps"].items()}
    w = {} if started else _writes(ev)
    if name == "route":
        steps["orchestrator"]["route"] = "active" if started else "done"
        if started:
            update_job(job_id, stage=STEP_LABELS["route"])
        else:
            r = w.get("route", {})
            how = {"caller": "chosen by you", "images": "the request contains images",
                   "llm": "interpreted by the language model", "no_router": "no language model to interpret it",
                   "router_error": "the language model failed"}.get(r.get("by"), r.get("by"))
            intent = r.get("intent")
            label = {"predict": "image classification", "train": "training campaign",
                     "models": "question about the models", "question": "question",
                     "unclear": "unclear"}.get(intent, intent)
            details = [f"reason: {r.get('reason', '')}"]
            if r.get("extracted"):
                details.append("constraints read from the request: "
                               + ", ".join(f"{k}={v}" for k, v in r["extracted"].items()))
            activity(job_id, "Orchestrator", "route", f"Request read as: {label} ({how}).", details)
            with JOBS_LOCK:
                if intent == "train" and job.get("lane") == "train":
                    job["view"], job["campaign"] = "train", True
                elif intent == "models":
                    _as_models_job(job, graph, steps)
                elif intent == "question":
                    job["view"] = "answer"
    elif name == "clarify":
        if started:
            graph["human"] = "waiting"
            activity(job_id, "Orchestrator", "clarify", "The request could be read in several ways. Asking you.")
        elif not ev.get("interrupts"):
            graph["human"] = "done"
            r = w.get("route", {})
            with JOBS_LOCK:
                if r.get("intent") == "train" and job.get("lane") == "train":
                    job["view"], job["campaign"] = "train", True
                elif r.get("intent") == "models":
                    _as_models_job(job, graph, steps)
                elif r.get("intent") == "question":
                    job["view"] = "answer"
    elif name == "answer" and not started:
        activity(job_id, "Orchestrator", "answer", w.get("report", ""))
    elif name == "explain":
        if started:
            update_job(job_id, stage=STEP_LABELS["explain"])
        else:
            activity(job_id, "Orchestrator", "explain", "Answer written.",
                     level="error" if w.get("error") else "")
    elif name == "inspector":
        if started:
            with JOBS_LOCK:
                _as_models_job(job, graph, steps)
            if not job.get("_inspector_announced"):
                job["_inspector_announced"] = True
                activity(job_id, "Orchestrator", "inspector", "Sent the question to the Models agent.")
        graph["inspector"] = "active" if started else ("error" if ev.get("error") else "done")
    elif name == "trainer":
        if not started and ev.get("interrupts"):
            return                     # the subgraph paused on a human approval: not finished
        graph["trainer"] = "active" if started else ("error" if ev.get("error") else "done")
        if started and not job.get("_trainer_announced"):
            # LangGraph re-runs the parent node on every resume: announce the handoff once
            job["_trainer_announced"] = True
            activity(job_id, "Orchestrator", "trainer", "Sent the request to the Training agent.")
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
        activity(job_id, "Training agent", step, f"Stopped: {ev['error']}", level="error")
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
        job["stage"] = (f"Training {data['run']}: epoch {data.get('epoch')}, "
                        f"validation balanced accuracy {float(data.get('val_balanced_acc') or 0):.3f}")


def _train_payload(values: dict) -> dict:
    tr = values.get("train_result") or {}
    return {"kind": "train", "report": values.get("report", ""), "route": values.get("route"),
            "campaign_id": tr.get("campaign_id"), "plan": tr.get("plan"),
            "stop_reason": tr.get("stop_reason"),
            "trials": [_compact_trial(t) for t in tr.get("trials", [])],
            "candidate": {k: (tr.get("candidate") or {}).get(k) for k in ("run", "arch", "best_score", "dir")},
            "evaluation": tr.get("evaluation") or {}, "promotion": tr.get("promotion") or {}}


def _as_models_job(job: dict, graph: dict, steps: dict) -> None:
    """A text job routed to the Models agent: show that agent, not the trainer."""
    job["view"] = "models"
    graph.update({"inspector": graph.get("inspector", "idle"), "trainer": "skipped",
                  "tester": "skipped", "reviewer": "skipped"})
    if graph.get("human") == "idle":
        graph["human"] = "skipped"
    steps.setdefault("inspector", {s: "idle" for s in MODELS_STEPS})
    steps.setdefault("orchestrator", {})["finalize"] = "skipped"


def _models_payload(values: dict) -> dict:
    mr = values.get("models_result") or {}
    facts = mr.get("facts") or {}
    return {"kind": "models", "report": values.get("report", ""), "error": values.get("error", ""),
            "answered_by": mr.get("answered_by"), "claims": mr.get("claims"), "llm_error": mr.get("llm_error", ""),
            "question": mr.get("request", ""),
            "validation_set": facts.get("validation_set"), "ensemble": facts.get("ensemble"),
            "models": facts.get("models", []), "caveats": facts.get("caveats", []),
            "charts": mr.get("charts") or {}}


def _text_payload(values: dict) -> dict:
    if values.get("models_result"):
        return _models_payload(values)
    if values.get("train_result"):
        return _train_payload(values)
    return {"kind": "answer", "report": values.get("report", ""), "route": values.get("route"),
            "error": values.get("error", "")}


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
            "vote_class": f["vote_class"], "vote_probability": f.get("vote_probability"),
            "runner_up_class": f.get("runner_up_class"), "margin": f.get("margin"),
            "rationale": f["rationale"],
            "verdict": r.get("verdict", ""), "flagged": any(x["severity"] != "info" for x in issues),
            "user_summary": r.get("user_summary", ""),
            "decisive_model_explanation": r.get("decisive_model_explanation", ""),
            "decisive": {k: d.get(k) for k in ("model", "share_of_final_class_score",
                                               "strongest_model", "strongest_model_agrees")},
            "soft_vote": dict(list(v["soft_vote"].items())[:5]),
            "issues": issues,
            "sources": f.get("sources", []),
            "past": f.get("past_predictions", [])[:1],
            "resized_from": f.get("resized_from"),
        })
    return {
        "kind": "predict",
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
        settings, config, lane = job["settings"], job["config"], job["lane"]
    as_payload = _result_payload if lane == "predict" else _text_payload
    lock = {"predict": RUN_LOCK, "train": TRAIN_LOCK}.get(lane)
    try:
        if lock is not None:
            lock.acquire()
        try:
            app = get_app(settings, lane=lane)
            for ns, mode, ev in app.stream(payload, config, stream_mode=["tasks", "custom"], subgraphs=True):
                if mode == "custom":
                    on_custom_event(job_id, ev)
                else:
                    on_task_event(job_id, ns, ev)
            snap = app.get_state(config)
        finally:
            if lock is not None:
                lock.release()
        pending = [i for t in snap.tasks for i in (t.interrupts or [])]
        if pending:
            with JOBS_LOCK:
                graph = dict(JOBS[job_id]["graph"])
                graph["human"] = "waiting"
                steps = {k: dict(v) for k, v in JOBS[job_id]["steps"].items()}
                steps["human"]["decision"] = "active"
            update_job(job_id, status="waiting_human", interrupt=pending[0].value,
                       graph=graph, steps=steps, result=as_payload(snap.values),
                       stage="Waiting for your decision")
            return
        values = snap.values
        with JOBS_LOCK:
            graph = dict(JOBS[job_id]["graph"])
        failed = bool(values.get("error"))
        graph["orchestrator"] = "error" if failed else "done"
        if graph.get("human") == "idle":
            graph["human"] = "skipped"
        if graph.get("trainer") == "active":
            graph["trainer"] = "done"
        if graph.get("inspector") == "active":
            graph["inspector"] = "done"
        activity(job_id, "Orchestrator", "done",
                 "Stopped: " + values["error"] if failed else "Finished.", level="error" if failed else "")
        update_job(job_id, status="error" if failed else "complete", percent=100,
                   stage="Stopped" if failed else "Finished",
                   graph=graph, result=as_payload(values), interrupt=None,
                   error=values.get("error", ""))
        HEALTH["time"] = 0.0                       # logs or models changed
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
        activity(job_id, "Orchestrator", "error", f"Stopped: {exc}", level="error")
        update_job(job_id, status="error", stage="Stopped", error=str(exc), graph=graph, steps=steps)


def _save_uploads(uploads: list[dict], folder: Path) -> list[Path]:
    paths = []
    for n, up in enumerate(uploads):
        encoded = str(up.get("data") or "")
        if "," not in encoded:
            raise ValueError("One of the uploads is not a valid image.")
        raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("Each image must be at most 10 MB.")
        name = Path(str(up.get("name") or f"image{n}.png")).name
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError("Use PNG, JPG, BMP or TIFF images.")
        stem = re.sub(r"[^\w.-]+", "_", Path(name).stem)[:60] or f"image{n}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{n + 1:02d}_{stem}{suffix}"
        path.write_bytes(raw)
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            path.unlink(missing_ok=True)
            raise ValueError(f"{name} is not a readable image.")
        paths.append(path)
    return paths


def _train_constraints(raw: dict | None) -> dict:
    from trainer.space import AUTONOMY, SELECT_ON
    raw, out = raw or {}, {}
    arch = str(raw.get("arch") or "").strip()
    if arch and arch != "auto":
        if arch not in _archs():
            raise ValueError(f"Unknown architecture '{arch}'. Choose from: {', '.join(_archs())}.")
        out["arch"] = arch
    names = {"max_trials": "The number of runs", "max_minutes": "The time budget",
             "max_epochs_per_run": "The epochs per run", "patience": "The patience"}
    for key, lo, hi, cast in (("max_trials", 1, 20, int), ("max_minutes", 1, 1440, float),
                              ("max_epochs_per_run", 1, 200, int), ("patience", 1, 50, int)):
        if raw.get(key) not in (None, ""):
            try:
                v = cast(float(raw[key]))
            except (TypeError, ValueError):
                raise ValueError(f"{names[key]} must be a number.")
            if not lo <= v <= hi:
                raise ValueError(f"{names[key]} must be between {lo} and {hi}.")
            out[key] = v
    if raw.get("target_score") not in (None, ""):
        v = float(raw["target_score"])
        if not 0 < v < 1:
            raise ValueError("The target must be between 0 and 1, e.g. 0.60.")
        out["target_score"] = v
    if raw.get("autonomy") in AUTONOMY:
        out["autonomy"] = raw["autonomy"]
    elif raw.get("autonomy"):
        raise ValueError(f"Autonomy must be one of: {', '.join(AUTONOMY)}.")
    if raw.get("select_on") in SELECT_ON:
        out["select_on"] = raw["select_on"]
    elif raw.get("select_on"):
        raise ValueError(f"The selection metric must be one of: {', '.join(SELECT_ON)}.")
    return out                                        # auto_approve_plan is never set from the web


# --- context for questions ----------------------------------------------------------

def _clip(text: str, n: int) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "..."


def _job_digest(job: dict) -> str:
    """The latest result, as facts the assistant can quote."""
    r = job.get("result") or {}
    if r.get("kind") == "predict":
        lines = [f"Latest prediction (execution {r.get('execution_id')}), testing agent: {r.get('tester')}, "
                 f"reviewer: {r.get('reviewer')}."]
        for img in r.get("images", []):
            votes = ", ".join(f"{k} {v:.2f}" for k, v in img["soft_vote"].items())
            lines.append(f"- {img['file']}: final class {img['final_class']}, confidence {img['confidence_level']}"
                         + (f", changed from the vote ({img['vote_class']})" if img["overridden"] else "")
                         + f". Soft vote: {votes}. Margin to runner-up: {img.get('margin')}."
                         + f" Reviewer verdict: {img.get('verdict') or 'n/a'}.")
            d = img.get("decisive") or {}
            if d.get("model"):
                lines.append(f"  Decisive model {d['model']}, {float(d.get('share_of_final_class_score') or 0):.0%} "
                             f"of the final class score; strongest model {d.get('strongest_model')} "
                             f"{'agrees' if d.get('strongest_model_agrees') else 'disagrees'}.")
            for i in img.get("issues", []):
                lines.append(f"  [{i['severity']}] {i['description']}")
            if img.get("user_summary") or img.get("rationale"):
                lines.append(f"  Summary: {_clip(img.get('user_summary') or img.get('rationale'), 600)}")
        if r.get("human_decision"):
            lines.append(f"Human decision: {r['human_decision'].get('decision')}.")
        return "\n".join(lines)
    if r.get("kind") == "train":
        ev = r.get("evaluation") or {}
        lines = [f"Latest training campaign {r.get('campaign_id')}: stopped because {r.get('stop_reason') or 'done'}."]
        for t in r.get("trials", []):
            lines.append(f"- {t['run']} ({t['action']}, {t['arch']}, lr {t['lr']}, class weight {t['class_weight']}, "
                         f"augmentation {t['augmentation']}): best score {t['best_score']}, "
                         f"stop reason {t['stop_reason']}. Diagnosis: {_clip(t.get('verdict'), 300)}")
        if ev.get("ensemble_now"):
            lines.append(f"Promotion check: ensemble balanced accuracy {ev['ensemble_now']['balanced_acc']:.4f}, "
                         f"with the candidate {ev['ensemble_with_candidate']['balanced_acc']:.4f}.")
        lines.append("Promoted: " + ((r.get("promotion") or {}).get("name") or "no"))
        return "\n".join(lines)
    return ""


def build_context(job_id: str | None, history: list | None) -> str:
    h = health()
    parts = ["SYSTEM STATE",
             f"Ensemble ({len(h['models'])} models): " + "; ".join(
                 f"{m['name']} (val balanced accuracy {m['val_balanced_acc']}, vote weight {m['skill']}"
                 + (f", never predicts: {', '.join(m['blind_classes'])}" if m.get("blind_classes") else "") + ")"
                 for m in h["models"]),
             f"Clinical knowledge base: {h['knowledge']['chunks']} passages from {h['knowledge']['sources']} sources.",
             f"Prediction log: {h['logs']['predictions']} entries; review log: {h['logs']['reviews']} entries.",
             f"Language models available: {', '.join(h['llm']['options']) or 'none'}.",
             f"Sample images: {', '.join(h['samples'])}.",
             f"Training campaign running now: {'yes' if h['campaign_running'] else 'no'}."]
    with JOBS_LOCK:
        job = JOBS.get(job_id or "")
        digest = _job_digest(job) if job else ""
    if digest:
        parts += ["", "LATEST RESULT", digest]
    turns = [t for t in (history or []) if isinstance(t, dict)][-6:]
    if turns:
        parts += ["", "RECENT CONVERSATION"]
        parts += [f"{'User' if t.get('role') == 'user' else 'Assistant'}: {_clip(t.get('text'), 1200)}" for t in turns]
    return "\n".join(parts)


def _new_job(kind: str, lane: str, settings: dict, **extra) -> tuple[str, dict]:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "lane": lane, "view": kind, "status": "running", "percent": 2,
           "stage": "Starting", "settings": settings, "config": {"configurable": {"thread_id": job_id}},
           "images": [], "activities": [], "result": None, "error": "", "interrupt": None, "contexts": {},
           "campaign": False, **extra}
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job_id, job


def start_text_job(payload: dict, settings: dict) -> str:
    """A training campaign (mode 'train'), the Models agent (mode 'models'), or free
    text the orchestrator must route (mode '')."""
    mode = payload.get("mode") if payload.get("mode") in ("train", "models") else ""
    request = str(payload.get("request") or "")[:2000]
    if mode == "models" and not request:
        request = "Which models are available, and how do they perform on the validation set?"
    constraints = _train_constraints(payload.get("train"))
    busy = _campaign_active() or TRAIN_LOCK.locked()
    # The Models agent only reads files: it never waits behind a campaign.
    lane = "train" if mode == "train" or (not busy and mode != "models") else "chat"
    job_id, job = _new_job(
        {"train": "train", "models": "models"}.get(mode, "text"), lane, settings,
        graph={"orchestrator": "active", "trainer": "idle", "human": "idle",
               "tester": "skipped", "reviewer": "skipped", "inspector": "idle"},
        steps={"trainer": {s: "idle" for s in TRAINER_STEPS}, "human": {"decision": "idle"},
               "orchestrator": {"route": "idle", "finalize": "skipped"}, "tester": {}, "reviewer": {},
               "inspector": {s: "idle" for s in MODELS_STEPS}},
        training={"plan": None, "max_trials": constraints.get("max_trials"), "trials_done": 0,
                  "trials": [], "runs": {}, "current": None, "campaign_id": None},
        campaign=mode == "train")
    tm = settings["trainer_model"] if settings["trainer_model"] is not None else settings["tester_model"]
    if mode == "train":
        activity(job_id, "Orchestrator", "start",
                 f"Training request received. Training agent: {tm or 'rule-based policy (no language model)'}.",
                 ["constraints: " + (", ".join(f"{k}={v}" for k, v in constraints.items())
                                     or "none, defaults apply and can be edited in the plan")])
        if busy:
            activity(job_id, "Orchestrator", "queue", "Another campaign is running. This one starts when it ends.")
    elif mode == "models":
        with JOBS_LOCK:
            _as_models_job(job, job["graph"], job["steps"])
        activity(job_id, "Orchestrator", "start",
                 f"Question about the models. Models agent: {settings['tester_model'] or 'summary computed in code (no language model)'}.")
    else:
        activity(job_id, "Orchestrator", "start", "Message received.")
    inputs = {"request": request, "mode": mode, "constraints": constraints}
    if mode != "train":
        inputs["context"] = build_context(payload.get("contextJob"), payload.get("history"))
    threading.Thread(target=run_stream, daemon=True, args=(job_id, inputs)).start()
    return job_id


def start_job(payload: dict) -> str:
    settings = normalize_settings(payload.get("settings"))
    has_images = bool(payload.get("samples") or payload.get("uploads"))
    if payload.get("mode") in ("train", "models") or (not has_images and str(payload.get("request") or "").strip()):
        return start_text_job(payload, settings)
    samples = [str(s) for s in payload.get("samples") or []]
    available = set(health()["samples"])
    if samples == ["*"]:
        samples = sorted(available)
    unknown = [s for s in samples if s not in available]
    if unknown:
        raise ValueError(f"Unknown sample: {', '.join(unknown)}.")
    job_id = uuid.uuid4().hex[:12]
    paths = [SAMPLES_DIR / s for s in samples]
    paths += _save_uploads(list(payload.get("uploads") or []), UPLOAD_ROOT / job_id)
    if not paths:
        raise ValueError("Attach at least one image or choose a sample.")
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"At most {MAX_IMAGES} images per request.")
    job = {
        "id": job_id, "kind": "predict", "lane": "predict", "view": "predict", "status": "running",
        "percent": 1, "stage": "Starting", "settings": settings,
        "config": {"configurable": {"thread_id": job_id}},
        "images": [{"file": p.name, "path": str(p)} for p in paths],
        "graph": {"orchestrator": "active", "tester": "idle", "reviewer": "idle",
                  "human": "idle" if settings["ask_human"] else "skipped"},
        "steps": {"tester": {s: "idle" for s in TESTER_STEPS},
                  "reviewer": {s: "idle" for s in REVIEWER_STEPS},
                  "human": {"decision": "idle" if settings["ask_human"] else "skipped"},
                  "orchestrator": {"route": "idle", "finalize": "idle"}},
        "activities": [], "result": None, "error": "", "interrupt": None, "contexts": {}, "campaign": False,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    who = settings["tester_model"] or "ensemble vote only"
    rv = (settings["reviewer_model"] if settings["reviewer_model"] is not None
          else settings["tester_model"]) or "automatic checks only"
    activity(job_id, "Orchestrator", "start",
             f"Received {_n(len(paths), 'image', 'images')}. Testing agent: {who}. Reviewer: {rv}.")
    if RUN_LOCK.locked():
        activity(job_id, "Orchestrator", "queue", "Another analysis is running. This one starts when it ends.")
    request = str(payload.get("request") or "")[:2000]
    threading.Thread(target=run_stream, daemon=True,
                     args=(job_id, {"image_paths": [str(p) for p in paths], "request": request})).start()
    return job_id


REVIEW_DECISIONS = {"accepted": "Accept the results", "rejected": "Reject the results",
                    "accepted_with_note": "Accept with a note"}


def resume_job(payload: dict) -> None:
    """Answer whatever the graph is waiting on. Each interrupt declares its own
    `options`; the reviewer's (which predates them) keeps its three decisions."""
    job_id = str(payload.get("job") or "")
    decision = str(payload.get("decision") or "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job["status"] != "waiting_human":
            raise ValueError("Nothing is waiting for a decision.")
        pending = job.get("interrupt") or {}
        options = {o["value"]: o["label"] for o in pending.get("options", [])} or REVIEW_DECISIONS
        if decision not in options:
            raise ValueError("This choice is not available for the current decision.")
        job["status"] = "running"
        job["interrupt"] = None
        job["graph"]["human"] = "active"
        if pending.get("kind") == "clarify_intent" and decision == "train" and job.get("lane") == "train":
            job["view"], job["campaign"] = "train", True
        if pending.get("kind") == "clarify_intent" and decision == "models":
            _as_models_job(job, job["graph"], job["steps"])
    note = str(payload.get("note") or "")[:1000]
    resume = {"decision": decision, "note": note}
    edits = payload.get("edits")
    if pending.get("kind") == "approve_plan" and isinstance(edits, dict):
        resume["edits"] = {k: v for k, v in edits.items() if k in set(pending.get("editable", []))}
    activity(job_id, "You", pending.get("kind") or "decision", f"Decision: {options[decision]}.",
             ([f"note: {note}"] if note else []) + ([f"edits: {resume['edits']}"] if resume.get("edits") else []))
    threading.Thread(target=run_stream, args=(job_id, Command(resume=resume)), daemon=True).start()


def public_job(job: dict) -> dict:
    return {k: v for k, v in job.items()
            if k not in ("config", "_files", "contexts", "_trainer_announced", "_inspector_announced")} | {
        "context_steps": sorted(job.get("contexts", {}))}


def knowledge_base():
    from tester.memory import KnowledgeBase
    with KB_LOCK:
        if "kb" not in KB:
            KB["kb"] = KnowledgeBase(CORPUS)
        return KB["kb"]


def kb_overview() -> dict:
    from tester.memory import CLASS_QUERIES
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
        "method": "TF-IDF on unigrams and bigrams (sublinear term frequency, English stop words), with the "
                  "source title prepended to each passage. The query is expanded with the dermoscopic terms "
                  "of the class. Passages are ranked by cosine similarity.",
    }


def kb_search(query: str, k: int) -> dict:
    query = query.strip()[:500]
    if not query:
        raise ValueError("Type something to search for.")
    kb = knowledge_base()
    hits = kb.search(query, k=max(1, min(k, 20)))
    full = {c["chunk_id"]: c["text"] for c in kb.chunks}
    return {"query": query, "results": [{**h, "text": full[h["chunk_id"]]} for h in hits]}


# --- isolated test evaluation ---------------------------------------------------------

EVALS: dict[str, dict] = {}
EVAL_LOCK = threading.Lock()


def _report_summary(r: dict) -> dict:
    return {k: r.get(k) for k in ("model", "arch", "evaluated_at", "test_dataset", "test_images", "img_size",
                                  "test_balanced_acc", "test_macro_f1", "test_macro_auc", "test_acc",
                                  "per_class_recall", "report_path")}


def evaluations() -> dict:
    reports = []
    if paths.EVALUATIONS.is_dir():
        for f in sorted(paths.EVALUATIONS.glob("*.json")):
            try:
                reports.append(_report_summary({**json.loads(f.read_text()), "report_path": str(f.relative_to(paths.ROOT))}))
            except (OSError, ValueError):
                pass
    models = [{"name": m["name"], "folder": m["folder"], "arch": m["arch"]} for m in health()["models"]]
    with JOBS_LOCK:
        running = {k: dict(v) for k, v in EVALS.items()}
    return {"models": models, "reports": reports, "runs": running, "classes": CLASS_NAMES}


def start_evaluation(payload: dict) -> dict:
    name = str(payload.get("model") or "")
    card = ModelZoo().cards.get(name)
    if card is None:
        raise ValueError(f"Unknown model '{name}'.")
    with JOBS_LOCK:
        if EVALS.get(name, {}).get("status") == "running":
            raise ValueError(f"{name} is already being evaluated.")
        EVALS[name] = {"status": "running", "started": time.time(), "error": "", "report": None}

    def work():
        from evaluate import evaluate_checkpoint
        try:
            with EVAL_LOCK:
                report = evaluate_checkpoint(card.path)
            report["report_path"] = str(Path(report["report_path"]).relative_to(paths.ROOT))
            with JOBS_LOCK:
                EVALS[name].update(status="done", report=_report_summary(report))
        except Exception as exc:
            traceback.print_exc()
            with JOBS_LOCK:
                EVALS[name].update(status="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


# --- providers ----------------------------------------------------------------------

def provider_action(path: str, payload: dict):
    if path == "/api/providers":
        entry = providers.save_provider(payload)
        _invalidate()
        return {"ok": True, "provider": entry}
    if path == "/api/providers/delete":
        providers.delete_provider(str(payload.get("id") or ""))
        _invalidate()
        return {"ok": True}
    if path == "/api/providers/key":
        pid = str(payload.get("id") or "")
        providers.get_provider(pid)
        providers.set_secret(pid, str(payload.get("key") or ""))
        _invalidate()
        return {"ok": True}
    if path == "/api/providers/models":
        return {"models": providers.fetch_models(str(payload.get("id") or ""))}
    if path == "/api/providers/test":
        spec = str(payload.get("model") or "")
        if not MODEL_RE.match(spec):
            raise ValueError("Choose a model to test.")
        return providers.test_model(spec)
    if path == "/api/providers/default":
        providers.set_default_model(str(payload.get("model") or ""))
        _invalidate()
        return {"ok": True}
    return None


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
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload) -> None:
        self.send_bytes(status, "application/json; charset=utf-8",
                        json.dumps(payload, ensure_ascii=False, default=str).encode())

    def _local_origin(self) -> bool:
        """Refuse writes from other web pages (CSRF) and from rebound host names."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]"):
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
            return False
        return (self.headers.get("Content-Type") or "").startswith("application/json")

    def read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES * MAX_IMAGES * 2:
            raise ValueError("The request is empty or too large.")
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
            self.send_json(200 if data else 404, data or {"error": "This analysis is no longer in memory."})
        elif parsed.path == "/api/kb":
            self.send_json(200, kb_overview())
        elif parsed.path == "/api/kb/search":
            try:
                self.send_json(200, kb_search(query.get("q", [""])[0], int(query.get("k", ["8"])[0] or 8)))
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
        elif parsed.path == "/favicon.ico":
            self.send_bytes(204, "image/x-icon", b"")
        elif parsed.path == "/api/health":
            self.send_json(200, health(force=query.get("refresh", ["0"])[0] == "1"))
        elif parsed.path == "/api/providers":
            self.send_json(200, providers.providers_overview())
        elif parsed.path == "/api/evaluations":
            self.send_json(200, evaluations())
        elif parsed.path == "/api/status":
            with JOBS_LOCK:
                job = JOBS.get(query.get("id", [""])[0])
                data = json.loads(json.dumps(public_job(job), default=str)) if job else None
            self.send_json(200 if data else 404, data or {"error": "This job is no longer in memory."})
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
        if not self._local_origin():
            self.send_json(403, {"error": "Requests are accepted only from the platform page on this machine."})
            return
        try:
            payload = self.read_payload()
            if self.path == "/api/run":
                self.send_json(202, {"job": start_job(payload)})
            elif self.path == "/api/resume":
                resume_job(payload)
                self.send_json(202, {"ok": True})
            elif self.path == "/api/evaluate":
                self.send_json(202, start_evaluation(payload))
            elif self.path.startswith("/api/providers"):
                out = provider_action(self.path, payload)
                self.send_json(200 if out is not None else 404, out if out is not None else {"error": "not found"})
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
