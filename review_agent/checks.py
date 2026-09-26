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
    parts = {name: cards[name]["skill"] * probs[idx][cls] for name, probs in predictions.items()}
    total = sum(parts.values()) or 1.0
    ranked = sorted(parts, key=parts.get, reverse=True)
    top = ranked[0]
    opinion = next(op for op in vote["per_model"] if op["model"] == top)
    strongest = max(models, key=lambda m: m["skill"])
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
    strongest = max(test["models"], key=lambda m: m["skill"])
    out = []

    for i, (f, v, ev) in enumerate(zip(test["final"], test["votes"], test["evidence"])):
        file, found = f["file"], []
        cls = f["final_class_id"]

        if test.get("llm_error"):
            found.append(_finding(file, "warning", "pipeline_error",
                                  "Il ragionamento LLM del tester è fallito: la risposta è il solo voto.",
                                  test["llm_error"]))
        if file in rejected:
            found.append(_finding(file, "warning", "vote_inconsistency",
                                  "Il tester ha proposto una classe che nessun modello sosteneva; "
                                  "il codice l'ha rifiutata e ha tenuto il voto."))
        if f["overridden"]:
            found.append(_finding(file, "warning", "vote_inconsistency",
                                  f"Il tester ha scavalcato il voto ({v['vote_class']} → {f['final_class']}): "
                                  "verificare che la motivazione lo giustifichi.",
                                  f"vote p={v['vote_probability']}, margin={v['margin']}"))
        if LEVELS[f["confidence_level"]] > LEVELS[v["confidence_level"]]:
            found.append(_finding(file, "warning", "vote_inconsistency",
                                  f"Confidenza dichiarata '{f['confidence_level']}' più alta di quella "
                                  f"calcolata dal voto ('{v['confidence_level']}').",
                                  f"margin={v['margin']}, agree={v['models_agreeing_with_vote']}"))
        if f["confidence_level"] == "low":
            found.append(_finding(file, "warning", "model_reliability",
                                  "Confidenza bassa: i modelli non convergono, serve una valutazione clinica.",
                                  f"margin={v['margin']}, agree={v['models_agreeing_with_vote']}"))

        top2 = {v["vote_class_id"], v["runner_up_class_id"]}
        if top2 == {MELANOMA, NEVUS} and v["margin"] < CLOSE_CALL_MARGIN:
            found.append(_finding(file, "critical", "model_reliability",
                                  "Melanoma e nevo sono le prime due classi con margine stretto: "
                                  "è la confusione più costosa, non va presentata come certa.",
                                  f"margin={v['margin']}"))
        if cls == MELANOMA or MELANOMA in top2:
            found.append(_finding(file, "info", "model_reliability",
                                  "Il melanoma è tra le classi in gioco: la decisione ha implicazioni cliniche."))

        for p in f.get("past_predictions", []):
            if p["final_class"] != f["final_class"]:
                same_models = set(p["models"]) == {m["name"] for m in test["models"]}
                found.append(_finding(file, "warning", "memory_conflict",
                                      f"In passato ({p['timestamp']}) la stessa immagine era stata "
                                      f"classificata {p['final_class']}."
                                      + ("" if same_models else " L'ensemble nel frattempo è cambiato."),
                                      f"modelli allora: {p['models']}"))
                break

        raw = raw_by_file.get(file, {})
        retrieved = {c["chunk_id"] for c in ev}
        invented = [c for c in raw.get("knowledge_used", []) if c not in retrieved]
        if invented:
            found.append(_finding(file, "warning", "unsupported_claim",
                                  "Il tester cita chunk della KB che non gli erano stati forniti.",
                                  f"{invented}"))
        m = _VISUAL_CLAIM.search(f.get("rationale", ""))
        if m:
            found.append(_finding(file, "critical", "unsupported_claim",
                                  "La motivazione descrive l'aspetto dell'immagine, ma il tester non la vede: "
                                  "l'affermazione non ha base.",
                                  f"«...{f['rationale'][max(0, m.start() - 40): m.end() + 40]}...»"))

        if cls in strongest["blind_classes"]:
            found.append(_finding(file, "info", "model_reliability",
                                  f"Il modello più forte ({strongest['name']}) non riconosce mai questa "
                                  "classe in validazione: la previsione poggia sui modelli più deboli."))
        if f.get("resized_from"):
            found.append(_finding(file, "warning", "input_quality",
                                  f"Immagine ridimensionata da {f['resized_from']} a 28x28: fuori "
                                  "distribuzione rispetto al training."))

        best_kb = max((c["score"] for c in reviewer_kb[i]
                       if c.get("about_class") == CLASS_NAMES[cls]), default=0.0)
        if best_kb < KB_WEAK_SCORE:
            found.append(_finding(file, "info", "kb_coverage",
                                  f"La knowledge base contiene poco su '{f['final_class']}': la verifica "
                                  "di coerenza con la letteratura è debole.",
                                  f"miglior score {best_kb:.3f}"))
        out.append(found)
    return out
