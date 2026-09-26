"""
The training agent as a LangGraph state machine.

    plan -> approve_plan -> propose -> validate -> review_proposal -> [approve_proposal]
                               ^         |  (invalid: back to propose, max 3)        |
                               |         v                                            v
                               |       stop ----------------------------------->  train
                               |                                                      |
                               +------------ decide <---------- analyse <-------------+
                                             | stop
                                             v
                              evaluate -> approve_promotion -> promote -> finish

Where the LLM is (optional - policy.py stands in without one):
    propose   what to run next: new run / warm restart / stop, architecture,
              hyper-parameters, augmentation with its dermoscopy rationale
    analyse   what the last segment shows, on top of facts computed in code

Where code decides:
    validate          bounds, alternatives, augmentation domain rules, warm-restart
                      shape, no repeated configuration (space.py)
    review_proposal   the training reviewer (review.py) + the autonomy gate
    train             one train.py segment; its early stop IS the trigger that
                      wakes the agent up when the metric stops improving
    decide            budget, number of runs, target, repeated failures
    evaluate          ensemble with vs without the candidate, on validation

Where a human decides (LangGraph interrupt(), resumed with Command(resume=...)):
    approve_plan       always - a campaign costs hours
    approve_proposal   when the gate says so (autonomy 'guarded': anything beyond
                       a small lr / weight-decay move on the same architecture)
    approve_promotion  always - nothing enters the prediction ensemble alone

Nothing is overwritten: every segment writes to runs_train/<campaign>/<run>/,
and a promoted model is COPIED to models_promoted/<new name>/.
"""

from __future__ import annotations

import json
import operator
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from fpvit.zoo import ARCHITECTURES, DEFAULT_ARCH
from predict_agent.memory import KnowledgeBase
from train_agent import policy
from train_agent.analysis import segment_facts
from train_agent.memory import ensemble_state, lessons, log_trial, past_runs
from train_agent.promotion import evaluate_candidate, promote
from train_agent.review import review_diagnosis, review_proposal
from train_agent.runner import RUNS_TRAIN, run_segment
from train_agent.space import (AUTONOMY, BOUNDS, SELECT_ON, config_signature, diff_configs, gate,
                               normalize_proposal, resolved_augmentation, validate_proposal)

AGENT_NAME = "derma_trainer"
KB_PATH = Path(__file__).with_name("training_corpus.json")
MAX_ATTEMPTS = 3
MAX_REJECTIONS = 3
MIN_SEGMENT_SECONDS = 60

TRAINER_STEPS = ["plan", "approve_plan", "propose", "validate", "review_proposal",
                 "approve_proposal", "train", "analyse", "decide", "evaluate",
                 "approve_promotion", "promote", "finish"]

DEFAULT_PLAN = {"select_on": "balanced_acc", "target_score": None, "max_trials": 4,
                "max_minutes": 60, "max_epochs_per_run": 30, "patience": 3,
                "autonomy": "guarded", "seed": 42, "num_workers": 2}
EDITABLE = {"arch", "select_on", "target_score", "max_trials", "max_minutes",
            "max_epochs_per_run", "patience", "autonomy"}

FLAG_QUERIES = {
    "plateau": "learning rate schedule warm restarts plateau convergence cosine",
    "overfitting": "overfitting regularization weight decay generalization gap augmentation cutout",
    "underfitting": "underfitting learning rate capacity training longer",
    "still_improving": "training longer schedule convergence",
    "collapsed_classes": "class imbalance minority classes class-balanced loss reweighting oversampling",
}
ARCH_QUERIES = {
    "fpvit": "vision transformer data augmentation regularization small datasets",
    "resnet18": "residual networks deep training",
    "efficientnet_b0": "efficientnet model scaling",
    "convnext_tiny": "convnet modernized design",
}


class TrainState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    request: str
    constraints: dict               # what the human / the orchestrator fixed
    plan: dict
    campaign_id: str
    memory: dict                    # deterministic memory, read once at plan time
    trials: Annotated[list[dict], operator.add]   # one per finished segment
    proposal: dict
    proposal_errors: list[str]
    proposal_warnings: list[str]
    attempts: int
    rejections: int
    human_feedback: str
    findings: list[dict]
    gate: dict
    approval: dict
    kb_context: list[dict]
    proposer_input: dict
    analyst_input: dict
    last_summary: dict
    seconds_spent: float
    stop_reason: str
    candidate: dict
    evaluation: dict
    promotion: dict
    report: str
    events: Annotated[list[dict], operator.add]
    error: str


def _ev(node: str, message: str, details: Optional[list[str]] = None, level: str = "") -> dict:
    return {"node": node, "message": message, "details": details or [], "level": level,
            "time": time.strftime("%H:%M:%S")}


def _runs_by_name(trials: list[dict]) -> dict[str, dict]:
    out = {}
    for t in trials:
        s, p = t["summary"], t["proposal"]
        out[s["run"]] = {"arch": p["arch"], "hparams": p["hparams"], "checkpoint": s.get("checkpoint"),
                         "best_score": s.get("best_score"), "signature": t.get("signature"),
                         "proposal": p, "summary": s}
    return out


def _reference(p: dict, trials: list[dict]) -> Optional[dict]:
    """What a proposal is compared with: its parent for a warm restart, else the last run."""
    if not trials:
        return None
    if p.get("action") == "warm_restart" and p.get("parent_run"):
        for t in trials:
            if t["summary"]["run"] == p["parent_run"]:
                return t["proposal"]
    return trials[-1]["proposal"]


def _compact_trial(t: dict) -> dict:
    s, p = t["summary"], t["proposal"]
    return {"run": s["run"], "action": p["action"], "arch": p["arch"], "parent_run": p.get("parent_run"),
            "hparams": p["hparams"], "augmentation": p["augmentation"],
            "status": s["status"], "best_score": s.get("best_score"), "best_epoch": s.get("best_epoch"),
            "epochs_completed": s.get("epochs_completed"), "stop_reason": s.get("stop_reason"),
            "flags": t["facts"].get("flags"), "verdict": (t.get("diagnosis") or {}).get("verdict")}


def _retrieve(kb: Optional[KnowledgeBase], queries: list[str], k: int = 2) -> list[dict]:
    if kb is None:
        return []
    seen, out = set(), []
    for q in queries:
        for c in kb.search(q, k=k, exclude=seen):
            seen.add(c["chunk_id"])
            out.append({k2: c[k2] for k2 in ("chunk_id", "title", "citation", "text", "score")} | {"query": q})
    return out


def build_trainer_graph(roles=None, kb_path: Optional[Path] = KB_PATH, checkpointer=None,
                        verbose: int = 0, runs_root: Path = RUNS_TRAIN):
    """Compile the training agent. `roles=None` uses the deterministic policy.

    `roles` is a train_agent.llm.LLMRoles. Compile with a checkpointer (or run it
    as a subgraph of a graph that has one): the human approvals are interrupts.
    """
    kb = KnowledgeBase(kb_path) if kb_path and Path(kb_path).exists() else None
    llm_name = roles.name if roles else "deterministic policy"

    def wrap(name, fn):
        def node(state):
            update = fn(state)
            if verbose:
                for e in update.get("events", []):
                    print(f"[trainer:{e['node']}] {e['message']}", file=sys.stderr, flush=True)
                    if verbose > 1:
                        for d in e["details"]:
                            print(f"    {d}", file=sys.stderr, flush=True)
            return update
        node.__name__ = name
        return node

    # --- plan ------------------------------------------------------------------
    def plan(state: TrainState) -> dict:
        c = {k: v for k, v in (state.get("constraints") or {}).items() if v not in (None, "")}
        arch = c.get("arch") or DEFAULT_ARCH
        if arch not in ARCHITECTURES:
            return {"error": f"architettura sconosciuta {arch!r}; scegli tra {sorted(ARCHITECTURES)}",
                    "stop_reason": "vincoli non validi"}
        p = {**DEFAULT_PLAN, **{k: c[k] for k in DEFAULT_PLAN if k in c},
             "arch": arch, "arch_fixed": bool(c.get("arch")), "llm": llm_name}
        if p["autonomy"] not in AUTONOMY:
            p["autonomy"] = "guarded"
        if p["select_on"] not in SELECT_ON:
            p["select_on"] = "balanced_acc"
        mem = {"ensemble": ensemble_state(), "past_runs": past_runs(12), "lessons": lessons(10)}
        weak = mem["ensemble"]["weak_classes"]
        p["goal"] = state.get("request") or (
            "Migliorare la balanced accuracy di validazione dell'ensemble di previsione"
            + (f", in particolare sulle classi deboli: {', '.join(weak)}" if weak else ""))
        campaign = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        details = [f"obiettivo: {p['goal']}",
                   f"architettura: {arch}" + (" (fissata dall'umano)" if p["arch_fixed"] else
                                              " (default; l'agente può sceglierne un'altra)"),
                   f"budget: {p['max_trials']} run, {p['max_minutes']} min, ≤{p['max_epochs_per_run']} "
                   f"epoche per run, trigger dopo {p['patience']} epoche senza miglioramento",
                   f"selezione su {p['select_on']}; autonomia {p['autonomy']}; LLM: {llm_name}",
                   f"ensemble attuale: {len(mem['ensemble']['models'])} modelli, migliore bal.acc "
                   f"{mem['ensemble']['best_val_balanced_acc']:.3f}; classi deboli: {', '.join(weak) or '—'}",
                   f"memoria: {len(mem['past_runs'])} run passati, {len(mem['lessons'])} lezioni"]
        return {"plan": p, "memory": mem, "campaign_id": campaign, "seconds_spent": 0.0,
                "attempts": 0, "rejections": 0, "error": "",
                "events": [_ev("plan", f"Piano della campagna {campaign}", details)]}

    def approve_plan(state: TrainState) -> dict:
        p = state["plan"]
        if (state.get("constraints") or {}).get("auto_approve_plan"):
            return {"approval": {"decision": "approve", "auto": True},
                    "events": [_ev("approve_plan", "Piano approvato in automatico (auto_approve_plan)")]}
        answer = interrupt({
            "kind": "approve_plan", "agent": AGENT_NAME,
            "question": "Approvi il piano di training? Puoi modificare budget, architettura e autonomia.",
            "plan": {k: p[k] for k in p if k not in ("seed", "num_workers")},
            "ensemble": state["memory"]["ensemble"],
            "editable": sorted(EDITABLE),
            "options": [{"value": "approve", "label": "Approva"}, {"value": "reject", "label": "Annulla"}],
        })
        answer = answer if isinstance(answer, dict) else {"decision": str(answer)}
        if answer.get("decision") not in ("approve", "approved", "accepted"):
            return {"approval": answer, "stop_reason": "piano non approvato",
                    "events": [_ev("approve_plan", "Piano rifiutato dall'umano", [answer.get("note", "")])]}
        edits = {k: v for k, v in (answer.get("edits") or {}).items() if k in EDITABLE}
        new = dict(p)
        for k, v in edits.items():
            if k == "arch":
                if v not in ARCHITECTURES:
                    continue
                new["arch"], new["arch_fixed"] = v, True
            elif k in ("autonomy",) and v in AUTONOMY or k == "select_on" and v in SELECT_ON:
                new[k] = v
            elif k in ("max_trials", "max_minutes", "max_epochs_per_run", "patience"):
                try:
                    new[k] = max(1, int(v))
                except (TypeError, ValueError):
                    pass
            elif k == "target_score":
                try:
                    new[k] = float(v) if v not in (None, "") else None
                except (TypeError, ValueError):
                    pass
        return {"plan": new, "approval": answer, "human_feedback": answer.get("note", ""),
                "events": [_ev("approve_plan", "Piano approvato" + (f" con modifiche {edits}" if edits else ""),
                               [answer.get("note", "")] if answer.get("note") else [])]}

    # --- propose / validate / review ------------------------------------------------
    def propose(state: TrainState) -> dict:
        p, trials = state["plan"], state.get("trials", [])
        last = trials[-1] if trials else None
        flags = (last or {}).get("facts", {}).get("flags", [])
        queries = [FLAG_QUERIES[f] for f in flags if f in FLAG_QUERIES]
        if not trials:
            queries += ["class imbalance skin lesion dermoscopy dataset",
                        ARCH_QUERIES.get(p["arch"], "")]
        queries.append("dermoscopy augmentation label-preserving colour orientation")
        kb_ctx = _retrieve(kb, [q for q in queries if q])
        used_s = state.get("seconds_spent", 0.0)
        payload = {
            "request": state.get("request", ""), "goal": p["goal"],
            "plan": {k: p[k] for k in ("arch", "arch_fixed", "select_on", "target_score", "max_epochs_per_run",
                                       "patience", "autonomy")},
            "budget_left": {"runs": p["max_trials"] - len(trials),
                            "minutes": round((p["max_minutes"] * 60 - used_s) / 60, 1)},
            "campaign_so_far": [_compact_trial(t) for t in trials],
            "last_segment_facts": (last or {}).get("facts"),
            "last_diagnosis": (last or {}).get("diagnosis"),
            "ensemble_now": {k: state["memory"]["ensemble"][k] for k in
                             ("best_val_balanced_acc", "best_recall_per_class", "weak_classes")},
            "past_runs": state["memory"]["past_runs"][:8],
            "lessons_from_past_campaigns": state["memory"]["lessons"],
            "human_feedback": state.get("human_feedback", ""),
            "your_previous_attempt_was_invalid": state.get("proposal_errors") or None,
            "knowledge": [{k: c[k] for k in ("chunk_id", "title", "citation", "text")} for c in kb_ctx],
        }
        source, level, err = llm_name, "", ""
        if roles is not None:
            try:
                raw = roles.propose(payload)
            except Exception as exc:              # never lose the campaign to an API error
                err = f"{type(exc).__name__}: {exc}"
                raw, source, level = policy.propose(p, trials), "deterministic policy (LLM failed)", "warn"
        else:
            raw = policy.propose(p, trials)
        prop = normalize_proposal(raw, _reference(raw, trials))
        asked = (raw.get("hparams") or {}).get("epochs")
        try:
            asked = int(asked) if asked is not None else None
        except (TypeError, ValueError):
            asked = None
        # epochs is only a ceiling (the plateau trigger ends a segment earlier): clamp it to
        # the plan instead of rejecting the whole proposal for it.
        prop["hparams"]["epochs"] = min(asked or p["max_epochs_per_run"], p["max_epochs_per_run"])
        if not trials and not p["arch_fixed"] and "arch" not in raw:
            prop["arch"] = p["arch"]
        msg = {"stop": "Propone di fermarsi", "new_run": f"Propone un nuovo run {prop['arch']}",
               "warm_restart": f"Propone un warm restart da {prop.get('parent_run')}"}[prop["action"]
               if prop["action"] in ("stop", "new_run", "warm_restart") else "new_run"]
        details = [f"da: {source}", f"motivazione: {prop['rationale']}"]
        if prop["action"] != "stop":
            details += [f"iperparametri: {prop['hparams']}",
                        f"augmentation: {prop['augmentation']}",
                        f"perché questa augmentation: {prop['augmentation_rationale']}"]
        if err:
            details.append(f"errore LLM: {err}")
        return {"proposal": prop, "proposer_input": payload, "kb_context": kb_ctx,
                "attempts": state.get("attempts", 0) + 1,
                "events": [_ev("propose", msg, details, level)]}

    def validate(state: TrainState) -> dict:
        prop, trials = state["proposal"], state.get("trials", [])
        errors, warnings = validate_proposal(prop, state["plan"], _runs_by_name(trials))
        if prop["action"] == "stop":
            return {"proposal_errors": [], "proposal_warnings": [],
                    "events": [_ev("validate", "Proposta di stop: nessun run da validare")]}
        return {"proposal_errors": errors, "proposal_warnings": warnings,
                "events": [_ev("validate", "Proposta valida" if not errors else
                               f"Proposta non valida ({len(errors)} errori): torna al proposer",
                               errors + [f"avviso: {w}" for w in warnings], "warn" if errors else "")]}

    def review_node(state: TrainState) -> dict:
        prop, trials = state["proposal"], state.get("trials", [])
        ref = _reference(prop, trials)
        facts = trials[-1]["facts"] if trials else None
        findings = review_proposal(prop, ref, facts, {c["chunk_id"] for c in state.get("kb_context", [])},
                                   state["memory"]["lessons"])
        findings += [{"severity": "warning", "category": "augmentation", "description": w, "evidence": ""}
                     for w in state.get("proposal_warnings", [])]
        needs, reasons = gate(prop, ref, state["plan"]["autonomy"], first=not trials, findings=findings)
        changes = diff_configs(prop, ref)
        details = [f"[{f['severity']}] {f['description']}" for f in findings] or ["nessun rilievo"]
        details.append("modifiche rispetto al riferimento: " + (", ".join(changes) or "—"))
        details += [f"serve un umano: {r}" for r in reasons]
        return {"findings": findings, "gate": {"needs_human": needs, "reasons": reasons,
                                               "changes": {k: v for k, v in changes.items()}},
                "events": [_ev("review_proposal",
                               ("Serve l'approvazione umana" if needs else "Approvata in automatico")
                               + f" (autonomia {state['plan']['autonomy']})", details,
                               "warn" if any(f["severity"] != "info" for f in findings) else "")]}

    def approve_proposal(state: TrainState) -> dict:
        prop = state["proposal"]
        try:
            aug = resolved_augmentation(prop["augmentation"])
        except (ValueError, TypeError):
            aug = prop["augmentation"]
        answer = interrupt({
            "kind": "approve_proposal", "agent": AGENT_NAME,
            "question": "Il training agent propone il prossimo run. Approvi?",
            "proposal": {k: prop[k] for k in ("action", "arch", "parent_run", "hparams", "rationale",
                                              "augmentation_rationale", "addresses", "expected_effect",
                                              "knowledge_used")},
            "augmentation_resolved": aug,
            "changes": state["gate"]["changes"], "why_human": state["gate"]["reasons"],
            "findings": state.get("findings", []),
            "options": [{"value": "approve", "label": "Approva"},
                        {"value": "reject", "label": "Chiedi un'altra proposta"},
                        {"value": "stop", "label": "Ferma la campagna"}],
        })
        answer = answer if isinstance(answer, dict) else {"decision": str(answer)}
        d = answer.get("decision", "")
        if d in ("approve", "approved", "accepted", "accepted_with_note"):
            return {"approval": answer, "human_feedback": "",
                    "events": [_ev("approve_proposal", "Proposta approvata dall'umano",
                                   [answer.get("note", "")] if answer.get("note") else [])]}
        if d == "stop":
            return {"approval": answer, "stop_reason": "fermata dall'umano",
                    "events": [_ev("approve_proposal", "Campagna fermata dall'umano")]}
        return {"approval": answer, "rejections": state.get("rejections", 0) + 1, "attempts": 0,
                "human_feedback": answer.get("note") or "L'umano ha rifiutato la proposta: proponi altro.",
                "events": [_ev("approve_proposal", "Proposta rifiutata: il proposer riceve il feedback",
                               [answer.get("note", "")])]}

    # --- train / analyse / decide ----------------------------------------------------
    def train(state: TrainState) -> dict:
        from langgraph.config import get_stream_writer
        try:
            writer = get_stream_writer()
        except Exception:
            writer = None
        p, prop, trials = state["plan"], state["proposal"], state.get("trials", [])
        remaining = p["max_minutes"] * 60 - state.get("seconds_spent", 0.0)
        share = remaining / max(p["max_trials"] - len(trials), 1)
        runs = _runs_by_name(trials)
        init_from = runs[prop["parent_run"]]["checkpoint"] if prop["action"] == "warm_restart" else None
        run_name = f"{prop['arch']}-t{len(trials) + 1:02d}" + ("-wr" if init_from else "")
        campaign_dir = runs_root / state["campaign_id"]

        def on_epoch(row):
            if writer:
                writer({"type": "epoch", "agent": AGENT_NAME, "campaign": state["campaign_id"], **row})

        started = f"Addestra {run_name} (≤{prop['hparams']['epochs']} epoche, budget {share / 60:.1f} min)"
        try:
            summary = run_segment(prop, p, campaign_dir, run_name, max(share, MIN_SEGMENT_SECONDS),
                                  init_from=init_from, on_epoch=on_epoch)
        except Exception as exc:
            summary = {"run": run_name, "status": "error", "arch": prop["arch"], "hparams": prop["hparams"],
                       "augmentation": prop["augmentation"], "checkpoint": None, "error": str(exc),
                       "seconds": 0.0, "dir": str(campaign_dir / run_name)}
        spent = state.get("seconds_spent", 0.0) + float(summary.get("seconds") or 0.0)
        if summary["status"] == "error":
            return {"last_summary": summary, "seconds_spent": spent,
                    "events": [_ev("train", f"{run_name}: errore", [summary.get("error", "")], "error")]}
        return {"last_summary": summary, "seconds_spent": spent, "events": [_ev(
            "train", f"{started}: {summary['stop_reason']} dopo {summary['epochs_completed']} epoche, "
                     f"{summary['selection_metric']} {summary['best_score']:.4f} all'epoca {summary['best_epoch']}",
            [f"recall per classe: {summary['per_class_recall']}",
             f"classi a recall 0: {', '.join(summary['collapsed_classes']) or '—'}",
             f"durata {summary['seconds']:.0f} s ({summary['mean_epoch_seconds']:.0f} s/epoca)",
             f"cartella: {summary['dir']}"])]}

    def analyse(state: TrainState) -> dict:
        s, prop, trials = state["last_summary"], state["proposal"], state.get("trials", [])
        prev = [t["summary"]["best_score"] for t in trials if t["summary"].get("best_score") is not None]
        parent = _runs_by_name(trials).get(prop.get("parent_run") or "", {}).get("summary") \
            if prop["action"] == "warm_restart" else None
        facts = segment_facts(s, max(prev) if prev else None, parent)
        kb_ctx = _retrieve(kb, [FLAG_QUERIES[f] for f in facts.get("flags", []) if f in FLAG_QUERIES], k=2)
        payload = {"request": state.get("request", ""), "facts": facts, "curve": s.get("curve"),
                   "run": {"arch": prop["arch"], "action": prop["action"], "hparams": prop["hparams"],
                           "augmentation": prop["augmentation"]},
                   "knowledge": [{k: c[k] for k in ("chunk_id", "title", "citation", "text")} for c in kb_ctx]}
        level = ""
        if s["status"] == "error":
            diag = {"verdict": f"Il run è fallito: {s.get('error', '')[:300]}", "recommendation": "correggere e riprovare"}
        elif roles is not None:
            try:
                diag = roles.analyse(payload)
            except Exception as exc:
                diag, level = policy.analyse(facts), "warn"
                diag["llm_error"] = f"{type(exc).__name__}: {exc}"
        else:
            diag = policy.analyse(facts)
        dfind = review_diagnosis(diag, facts) if s["status"] != "error" else []
        ref = _reference(prop, trials)
        trial = {"proposal": prop, "summary": s, "facts": facts, "diagnosis": diag,
                 "findings": state.get("findings", []) + dfind, "gate": state.get("gate", {}),
                 "approval": state.get("approval", {}), "signature": config_signature(prop),
                 "analyst_input": payload}
        delta = facts.get("improvement_vs_parent") if parent else facts.get("improvement_vs_campaign_best")
        log_trial({"kind": "trial", "campaign": state["campaign_id"], "run": s["run"], "arch": prop["arch"],
                   "action": prop["action"], "changes": {k: list(v) if isinstance(v, tuple) else v
                                                         for k, v in diff_configs(prop, ref).items()},
                   "diagnosis_before": (trials[-1]["facts"].get("flags") if trials else None),
                   "score": s.get("best_score"), "delta": delta, "status": s["status"],
                   "rationale": prop["rationale"]})
        return {"trials": [trial], "analyst_input": payload, "kb_context": kb_ctx, "attempts": 0,
                "events": [_ev("analyse", diag.get("verdict", ""),
                               [f"flag calcolati: {', '.join(facts.get('flags', [])) or '—'}",
                                f"raccomandazione: {diag.get('recommendation', '')}"]
                               + [f"[{f['severity']}] {f['description']}" for f in dfind], level)]}

    def decide(state: TrainState) -> dict:
        p, trials = state["plan"], state.get("trials", [])
        scores = [t["summary"]["best_score"] for t in trials if t["summary"].get("best_score") is not None]
        errors_in_row = 0
        for t in reversed(trials):
            if t["summary"]["status"] != "error":
                break
            errors_in_row += 1
        reason = ""
        if errors_in_row >= 2:
            reason = "due run consecutivi falliti"
        elif len(trials) >= p["max_trials"]:
            reason = f"raggiunto il numero massimo di run ({p['max_trials']})"
        elif p["max_minutes"] * 60 - state.get("seconds_spent", 0.0) < MIN_SEGMENT_SECONDS:
            reason = "budget di tempo esaurito"
        elif p.get("target_score") is not None and scores and max(scores) >= p["target_score"]:
            reason = f"obiettivo raggiunto ({max(scores):.4f} ≥ {p['target_score']})"
        return {"stop_reason": reason, "events": [_ev(
            "decide", f"Stop: {reason}" if reason else
            f"Continua: {len(trials)}/{p['max_trials']} run, "
            f"{state.get('seconds_spent', 0) / 60:.1f}/{p['max_minutes']} min")]}

    # --- evaluation and promotion -----------------------------------------------------
    def evaluate(state: TrainState) -> dict:
        ok = [t for t in state.get("trials", []) if t["summary"]["status"] != "error"
              and t["summary"].get("checkpoint") and t["summary"].get("best_score") is not None]
        if not ok:
            return {"candidate": {}, "evaluation": {},
                    "events": [_ev("evaluate", "Nessun candidato da valutare")]}
        best = max(ok, key=lambda t: t["summary"]["best_score"])["summary"]
        try:
            ev = evaluate_candidate(best)
        except Exception as exc:
            return {"candidate": best, "evaluation": {"error": str(exc)},
                    "events": [_ev("evaluate", f"Valutazione fallita: {exc}", level="error")]}
        return {"candidate": best, "evaluation": ev, "events": [_ev(
            "evaluate",
            f"Candidato {best['run']}: ensemble {ev['ensemble_now']['balanced_acc']:.4f} → "
            f"{ev['ensemble_with_candidate']['balanced_acc']:.4f} ({ev['delta_balanced_acc']:+.4f}) "
            + ("— migliora" if ev["improves"] else "— non migliora abbastanza"),
            [f"da solo: {ev['candidate_alone']['balanced_acc']:.4f}",
             "variazioni di recall: " + ", ".join(f"{c} {d:+.3f}" for c, d in ev["recall_changes"].items() if d),
             ev["caveat"]])]}

    def approve_promotion(state: TrainState) -> dict:
        cand, ev = state["candidate"], state["evaluation"]
        answer = interrupt({
            "kind": "approve_promotion", "agent": AGENT_NAME,
            "question": ("Il candidato migliora l'ensemble sulla validazione: lo promuovo?" if ev.get("improves")
                         else "Il candidato NON migliora l'ensemble in modo significativo. Promuoverlo comunque?"),
            "candidate": {k: cand.get(k) for k in ("run", "arch", "hparams", "best_score", "selection_metric",
                                                   "val_at_best", "per_class_recall", "dir")},
            "evaluation": ev, "recommendation": "promuovere" if ev.get("improves") else "non promuovere",
            "options": [{"value": "approve", "label": "Promuovi nell'ensemble"},
                        {"value": "reject", "label": "Non promuovere"}],
        })
        answer = answer if isinstance(answer, dict) else {"decision": str(answer)}
        ok = answer.get("decision") in ("approve", "approved", "accepted", "accepted_with_note")
        return {"approval": answer, "events": [_ev("approve_promotion",
                                                   "Promozione approvata" if ok else "Promozione rifiutata",
                                                   [answer.get("note", "")] if answer.get("note") else [])]}

    def promote_node(state: TrainState) -> dict:
        try:
            entry = promote(state["candidate"], state["evaluation"], state["campaign_id"], state["approval"])
        except Exception as exc:
            return {"promotion": {"error": str(exc)},
                    "events": [_ev("promote", f"Promozione fallita: {exc}", level="error")]}
        return {"promotion": entry, "events": [_ev(
            "promote", f"Promosso come models_promoted/{entry['name']}: vota dalla prossima previsione",
            [f"sha256 {entry['sha256'][:16]}…", f"sorgente {entry['source_run']} (non toccata)"])]}

    def finish(state: TrainState) -> dict:
        report = _report(state)
        campaign_dir = runs_root / state.get("campaign_id", "none")
        if campaign_dir.is_dir():
            record = {k: state.get(k) for k in ("campaign_id", "request", "constraints", "plan", "stop_reason",
                                                "candidate", "evaluation", "promotion", "seconds_spent")}
            record["trials"] = [{k: t[k] for k in ("proposal", "summary", "facts", "diagnosis", "findings",
                                                   "gate", "approval")} for t in state.get("trials", [])]
            record["report"] = report
            (campaign_dir / "campaign.json").write_text(json.dumps(record, indent=1, ensure_ascii=False,
                                                                   default=str))
        return {"report": report, "messages": [AIMessage(content=report, name=AGENT_NAME)],
                "events": [_ev("finish", "Campagna chiusa: " + (state.get("stop_reason") or "fine"))]}

    # --- routing ----------------------------------------------------------------------
    def after_plan(state):
        return "finish" if state.get("error") else "approve_plan"

    def after_approve_plan(state):
        return "finish" if state.get("stop_reason") else "propose"

    def after_validate(state):
        prop = state["proposal"]
        if prop["action"] == "stop":
            return "evaluate"
        if state.get("proposal_errors"):
            return "propose" if state.get("attempts", 0) < MAX_ATTEMPTS else "give_up"
        return "review_proposal"

    def after_review(state):
        return "approve_proposal" if state["gate"]["needs_human"] else "train"

    def after_approve_proposal(state):
        if state.get("stop_reason"):
            return "evaluate"
        if state.get("approval", {}).get("decision") in ("approve", "approved", "accepted", "accepted_with_note"):
            return "train"
        return "propose" if state.get("rejections", 0) < MAX_REJECTIONS else "evaluate"

    def after_decide(state):
        return "evaluate" if state.get("stop_reason") else "propose"

    def after_evaluate(state):
        ev = state.get("evaluation") or {}
        return "approve_promotion" if state.get("candidate") and "ensemble_now" in ev else "finish"

    def after_approve_promotion(state):
        ok = state.get("approval", {}).get("decision") in ("approve", "approved", "accepted", "accepted_with_note")
        return "promote" if ok else "finish"

    def give_up(state):
        return {"stop_reason": f"{MAX_ATTEMPTS} proposte non valide consecutive",
                "events": [_ev("validate", "Troppe proposte non valide: la campagna si ferma", level="error")]}

    g = StateGraph(TrainState)
    for name, fn in [("plan", plan), ("approve_plan", approve_plan), ("propose", propose),
                     ("validate", validate), ("review_proposal", review_node),
                     ("approve_proposal", approve_proposal), ("train", train), ("analyse", analyse),
                     ("decide", decide), ("evaluate", evaluate), ("approve_promotion", approve_promotion),
                     ("promote", promote_node), ("finish", finish), ("give_up", give_up)]:
        g.add_node(name, wrap(name, fn))
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["approve_plan", "finish"])
    g.add_conditional_edges("approve_plan", after_approve_plan, ["propose", "finish"])
    g.add_edge("propose", "validate")
    g.add_conditional_edges("validate", after_validate, ["propose", "review_proposal", "evaluate", "give_up"])
    g.add_edge("give_up", "evaluate")
    g.add_conditional_edges("review_proposal", after_review, ["approve_proposal", "train"])
    g.add_conditional_edges("approve_proposal", after_approve_proposal, ["train", "propose", "evaluate"])
    g.add_edge("train", "analyse")
    g.add_edge("analyse", "decide")
    g.add_conditional_edges("decide", after_decide, ["propose", "evaluate"])
    g.add_conditional_edges("evaluate", after_evaluate, ["approve_promotion", "finish"])
    g.add_conditional_edges("approve_promotion", after_approve_promotion, ["promote", "finish"])
    g.add_edge("promote", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer, name=AGENT_NAME)


def _report(state: dict) -> str:
    p = state.get("plan") or {}
    lines = [f"Campagna di training {state.get('campaign_id', '')}",
             f"Obiettivo: {p.get('goal', state.get('request', ''))}",
             f"Chiusura: {state.get('stop_reason') or state.get('error') or 'fine'}", ""]
    for t in state.get("trials", []):
        s, pr = t["summary"], t["proposal"]
        score = f"{s['best_score']:.4f}" if s.get("best_score") is not None else "—"
        lines.append(f"• {s['run']} ({pr['action']}, {pr['arch']}, lr {pr['hparams']['lr']:.2g}, "
                     f"class_weight {pr['hparams']['class_weight']}, aug '{pr['augmentation']['preset']}'): "
                     f"{s.get('selection_metric', '')} {score} — {s.get('stop_reason', s.get('error', ''))}")
        lines.append(f"  perché: {pr['rationale']}")
        if t.get("diagnosis", {}).get("verdict"):
            lines.append(f"  diagnosi: {t['diagnosis']['verdict']}")
    ev, cand = state.get("evaluation") or {}, state.get("candidate") or {}
    if "ensemble_now" in ev:
        lines += ["", f"Candidato {cand.get('run')}: ensemble di validazione "
                      f"{ev['ensemble_now']['balanced_acc']:.4f} → {ev['ensemble_with_candidate']['balanced_acc']:.4f} "
                      f"({ev['delta_balanced_acc']:+.4f})."]
    promo = state.get("promotion") or {}
    if promo.get("name"):
        lines.append(f"Promosso come models_promoted/{promo['name']}: il Testing agent lo usa da subito.")
    elif cand:
        lines.append("Nessun modello promosso: l'ensemble di previsione è invariato.")
    lines += ["", "Test split mai usato: la valutazione finale resta un passo manuale (evaluate_test.py)."]
    return "\n".join(lines)
