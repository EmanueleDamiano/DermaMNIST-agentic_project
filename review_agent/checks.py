"""
Deterministic checks on a finished prediction run. No LLM, no framework.

These are the facts the reviewer does not have to trust a model for: they are
computed from the predictor's own state (votes, models, memory, LLM output,
retrieval) and are passed to the reviewing LLM as ground truth, and shown to
the user whether or not an LLM review ran.

Each finding: {file, severity, category, description, evidence}
    severity  critical | warning | info
    category  vote_inconsistency | memory_conflict | unsupported_claim |
              model_reliability | input_quality | kb_coverage | pipeline_error
"""

from __future__ import annotations

import re

from predict_agent.models import CLASS_NAMES

LEVELS = {"low": 0, "medium": 1, "high": 2}
MELANOMA, NEVUS = CLASS_NAMES.index("melanoma"), CLASS_NAMES.index("melanocytic nevi")
CLOSE_CALL_MARGIN = 0.20
# Below this TF-IDF score the knowledge base has essentially nothing on a class.
KB_WEAK_SCORE = 0.10

# The reasoning LLM never sees the image: a sentence describing what the image
# shows is invented. Deliberately narrow phrases, to keep false alarms rare.
_VISUAL_CLAIM = re.compile(
    r"(l['’]immagine|la lesione|the image|the lesion)\s+(mostra|presenta|evidenzia|shows|displays|exhibits)"
    r"|\bsi\s+(osserva|osservano|vede|vedono|nota|notano)\b"
    r"|\b(is|are)\s+(visible|seen|observed)\b|\bcan be seen\b",
    re.IGNORECASE,
)


def _finding(file, severity, category, description, evidence=""):
    return {"file": file, "severity": severity, "category": category,
            "description": description, "evidence": evidence}


def decisive_model(final: dict, vote: dict, predictions: dict, idx: int,
                   models: list[dict]) -> dict:
    """Which model carried the final class, and by how much.

    The final class's soft-vote score is sum_m skill_m * p_m(class) / sum skill;
    each model's share of that sum is its contribution. The largest share is
    the 'decisive' model - the closest thing an ensemble has to "the model
    that made the prediction".
    """
    cls = final["final_class_id"]
    cards = {m["name"]: m for m in models}
    # Only the models that voted on this image: a model can sit out an image
    # it would have to upsample (predict_agent.graph.resize_info).
    voters = {op["model"] for op in vote["per_model"]}
    parts = {name: cards[name]["skill"] * probs[idx][cls] for name, probs in predictions.items()
             if name in voters}
    total = sum(parts.values()) or 1.0
    ranked = sorted(parts, key=parts.get, reverse=True)
    top = ranked[0]
    opinion = next(op for op in vote["per_model"] if op["model"] == top)
    strongest = max((m for m in models if m["name"] in voters), key=lambda m: m["skill"])
    strongest_op = next(op for op in vote["per_model"] if op["model"] == strongest["name"])
    return {
        "model": top,
        "share_of_final_class_score": round(parts[top] / total, 3),
        "probability_for_final_class": round(predictions[top][idx][cls], 4),
        "its_own_top_class": opinion["predicted_class"],
        "skill": cards[top]["skill"],
        "val_balanced_acc": cards[top]["val_balanced_acc"],
        "val_precision_for_final_class": round(cards[top]["val_per_class_precision"][cls], 4),
        "val_recall_for_final_class": round(cards[top]["val_per_class_recall"][cls], 4),
        "epoch": cards[top]["epoch"],
        "optimizer": cards[top]["optimizer"],
        "class_weight": cards[top]["class_weight"],
        "contributions": {n: round(parts[n] / total, 3) for n in ranked},
        "strongest_model": strongest["name"],
        "strongest_model_agrees": strongest_op["predicted_class_id"] == cls,
        "strongest_model_top_class": strongest_op["predicted_class"],
    }


def run_checks(test: dict, reviewer_kb: list[list[dict]]) -> list[list[dict]]:
    """One list of findings per image, aligned with test['final']."""
    raw_by_file = {d["file"]: d for d in (test.get("llm_output") or {}).get("decisions", [])}
    rejected = {r["file"] for r in test.get("rejected_overrides", [])}
    out = []

    for i, (f, v, ev) in enumerate(zip(test["final"], test["votes"], test["evidence"])):
        file, found = f["file"], []
        cls = f["final_class_id"]
        voters = {op["model"] for op in v["per_model"]}
        strongest = max((m for m in test["models"] if m["name"] in voters), key=lambda m: m["skill"])

        if test.get("llm_error"):
            found.append(_finding(file, "warning", "pipeline_error",
                                  "The tester's LLM reasoning failed: the answer is the vote alone.",
                                  test["llm_error"]))
        if file in rejected:
            found.append(_finding(file, "warning", "vote_inconsistency",
                                  "The tester proposed a class no model supported; "
                                  "the code rejected it and kept the vote."))
        if f["overridden"]:
            found.append(_finding(file, "warning", "vote_inconsistency",
                                  f"The tester overrode the vote ({v['vote_class']} → {f['final_class']}): "
                                  "check that the rationale justifies it.",
                                  f"vote p={v['vote_probability']}, margin={v['margin']}"))
        if LEVELS[f["confidence_level"]] > LEVELS[v["confidence_level"]]:
            found.append(_finding(file, "warning", "vote_inconsistency",
                                  f"Stated confidence '{f['confidence_level']}' is higher than the one "
                                  f"computed by the vote ('{v['confidence_level']}').",
                                  f"margin={v['margin']}, agree={v['models_agreeing_with_vote']}"))
        if f["confidence_level"] == "low":
            found.append(_finding(file, "warning", "model_reliability",
                                  "Low confidence: the models do not converge, a clinical assessment is needed.",
                                  f"margin={v['margin']}, agree={v['models_agreeing_with_vote']}"))

        top2 = {v["vote_class_id"], v["runner_up_class_id"]}
        if top2 == {MELANOMA, NEVUS} and v["margin"] < CLOSE_CALL_MARGIN:
            found.append(_finding(file, "critical", "model_reliability",
                                  "Melanoma and nevus are the top two classes with a narrow margin: "
                                  "the costliest confusion, it must not be presented as certain.",
                                  f"margin={v['margin']}"))
        if cls == MELANOMA or MELANOMA in top2:
            found.append(_finding(file, "info", "model_reliability",
                                  "Melanoma is among the classes in play: the decision has clinical implications."))

        for p in f.get("past_predictions", []):
            if p["final_class"] != f["final_class"]:
                same_models = set(p["models"]) == {m["name"] for m in test["models"]}
                found.append(_finding(file, "warning", "memory_conflict",
                                      f"Previously ({p['timestamp']}) the same image was "
                                      f"classified as {p['final_class']}."
                                      + ("" if same_models else " The ensemble has changed since then."),
                                      f"models then: {p['models']}"))
                break

        raw = raw_by_file.get(file, {})
        retrieved = {c["chunk_id"] for c in ev}
        invented = [c for c in raw.get("knowledge_used", []) if c not in retrieved]
        if invented:
            found.append(_finding(file, "warning", "unsupported_claim",
                                  "The tester cites KB chunks it was not given.",
                                  f"{invented}"))
        m = _VISUAL_CLAIM.search(f.get("rationale", ""))
        if m:
            found.append(_finding(file, "critical", "unsupported_claim",
                                  "The rationale describes how the image looks, but the tester cannot see it: "
                                  "the claim has no basis.",
                                  f"«...{f['rationale'][max(0, m.start() - 40): m.end() + 40]}...»"))

        if cls in strongest["blind_classes"]:
            found.append(_finding(file, "info", "model_reliability",
                                  f"The strongest model ({strongest['name']}) never recognises this "
                                  "class on validation: the prediction rests on the weaker models."))
        if f.get("all_upsampled"):
            found.append(_finding(file, "warning", "input_quality",
                                  f"Image {f['resized_from'][0]}x{f['resized_from'][1]} is smaller than the "
                                  "input of every model: all votes are out of distribution.",
                                  f.get("resize_note", "")))
        elif f.get("excluded_from_vote"):
            found.append(_finding(file, "info", "input_quality",
                                  f"Image {f['resized_from'][0]}x{f['resized_from'][1]} is too small for "
                                  f"{', '.join(f['excluded_from_vote'])}, which did not vote: the decision "
                                  "rests on the lower-resolution models only.",
                                  f.get("resize_note", "")))
        elif f.get("resized_from"):
            found.append(_finding(file, "info", "input_quality",
                                  f"Image {f['resized_from'][0]}x{f['resized_from'][1]} is resized to "
                                  "the models' inputs.", f.get("resize_note", "")))

        best_kb = max((c["score"] for c in reviewer_kb[i]
                       if c.get("about_class") == CLASS_NAMES[cls]), default=0.0)
        if best_kb < KB_WEAK_SCORE:
            found.append(_finding(file, "info", "kb_coverage",
                                  f"The knowledge base holds little on '{f['final_class']}': the "
                                  "consistency check against the literature is weak.",
                                  f"best score {best_kb:.3f}"))
        out.append(found)
    return out
