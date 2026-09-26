"""
The training reviewer: deterministic checks on what the LLM roles produced.

Same idea as review_agent/checks.py for predictions: facts computed in code,
each finding {severity, category, description, evidence}. A 'critical' or
'warning' finding forces the proposal in front of a human, whatever the
autonomy level (space.gate).
"""

from __future__ import annotations

from train_agent.space import diff_configs, resolved_augmentation

ISSUE_FLAGS = {"plateau", "overfitting", "underfitting", "still_improving", "collapsed_classes"}


def _f(severity, category, description, evidence=""):
    return {"severity": severity, "category": category, "description": description, "evidence": evidence}


def review_proposal(p: dict, ref: dict | None, facts: dict | None, retrieved_ids: set[str],
                    lessons: list[dict]) -> list[dict]:
    out = []
    if p["action"] == "stop":
        return out
    flags = set((facts or {}).get("flags", []))

    # 1. it addresses what the diagnosis actually shows
    for claim in p.get("addresses", []):
        if claim in ISSUE_FLAGS and facts and claim not in flags:
            out.append(_f("warning", "unsupported_claim",
                          f"la proposta dice di correggere '{claim}', ma i fatti del run non lo mostrano",
                          f"flag calcolati: {sorted(flags) or 'nessuno'}"))

    # 2. one change at a time (a campaign of 5 runs changing 4 things each teaches nothing)
    changes = diff_configs(p, ref)
    n = len(changes) - (1 if "epochs" in changes else 0)
    if ref and n > 2:
        out.append(_f("warning", "too_many_changes",
                      f"cambia {n} cose insieme: se il risultato cambia non si saprà perché",
                      ", ".join(sorted(changes))))

    # 3. augmentation coherent with the diagnosis
    try:
        new_aug = resolved_augmentation(p["augmentation"])
        old_aug = resolved_augmentation(ref["augmentation"]) if ref else None
    except (ValueError, TypeError, KeyError):
        new_aug, old_aug = None, None
    if new_aug and old_aug:
        stronger = (new_aug.get("cutout") and not old_aug.get("cutout")) or \
                   new_aug.get("rrc_scale_min", 1) < old_aug.get("rrc_scale_min", 1) or \
                   new_aug.get("rotation_degrees", 0) > old_aug.get("rotation_degrees", 0)
        if stronger and "underfitting" in flags:
            out.append(_f("warning", "augmentation",
                          "augmentation più forte mentre il modello è in underfitting: rende il "
                          "problema più difficile invece di più facile"))
        if stronger and facts and "overfitting" not in flags:
            out.append(_f("info", "augmentation",
                          "augmentation più forte senza overfitting misurato: è un regolarizzatore, "
                          "da giustificare"))
    if new_aug and new_aug.get("color_jitter") and new_aug.get("cj_hue", 0) > 0.03:
        out.append(_f("warning", "augmentation",
                      "hue jitter alto su immagini dermoscopiche: il colore è un segnale diagnostico",
                      f"cj_hue={new_aug['cj_hue']}"))

    # 4. the prose matches the fields (a choice written only in the rationale is not executed)
    import re
    prose = (p.get("rationale", "") + " " + p.get("augmentation_rationale", "")).lower()

    def mentioned(word):
        return re.search(rf"\b{re.escape(word)}\b", prose) is not None

    chosen_cw = p["hparams"].get("class_weight")
    other_cw = [w for w in ("inverse", "effective") if w != chosen_cw and mentioned(w)]
    if other_cw and not mentioned(str(chosen_cw)):
        out.append(_f("warning", "prose_vs_fields",
                      f"la motivazione parla di class_weight '{other_cw[0]}' ma il campo eseguito è "
                      f"'{chosen_cw}'", "solo i campi vengono eseguiti"))
    chosen_pre = p["augmentation"]["preset"]
    other_pre = [w for w in ("dihedral", "strong", "paper") if w != chosen_pre and mentioned(w)]
    if other_pre and not mentioned(chosen_pre):
        out.append(_f("warning", "prose_vs_fields",
                      f"la motivazione parla dell'augmentation '{other_pre[0]}' ma il preset eseguito è "
                      f"'{chosen_pre}'", "solo i campi vengono eseguiti"))

    # 5. citations are real
    invented = [c for c in p.get("knowledge_used", []) if c not in retrieved_ids]
    if invented:
        out.append(_f("warning", "unsupported_claim",
                      "cita passaggi della knowledge base che non gli erano stati forniti", str(invented)))

    # 6. repeating a change that made things worse before
    def direction(pair):
        old, new = pair
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool):
            return "up" if new > old else "down" if new < old else "same"
        return f"→{new}"

    for les in lessons:
        if les.get("arch") != p["arch"] or (les.get("delta") or 0) >= -0.01:
            continue
        same = [k for k, pair in (les.get("changes") or {}).items()
                if k in changes and k != "augmentation" and isinstance(pair, (list, tuple))
                and direction(pair) == direction(changes[k])]
        if same:
            out.append(_f("info", "memory",
                          f"in {les['campaign']}/{les['run']} lo stesso cambiamento ({', '.join(same)}) "
                          f"aveva peggiorato il punteggio di {les['delta']}"))
            break
    return out


def review_diagnosis(diag: dict, facts: dict) -> list[dict]:
    """The analyst LLM against the computed facts. The facts win."""
    out = []
    flags = set(facts.get("flags", []))
    for key, flag in (("overfitting", "overfitting"), ("underfitting", "underfitting"),
                      ("plateau", "plateau")):
        said = diag.get(key)
        if said is True and flag not in flags:
            out.append(_f("warning", "diagnosis",
                          f"l'analista dice '{key}' ma i numeri non lo mostrano: vale il calcolo",
                          f"gap {facts.get('gap_at_best')}→{facts.get('gap_at_end')}, "
                          f"stop_reason {facts.get('stop_reason')}"))
        if said is False and flag in flags:
            out.append(_f("info", "diagnosis",
                          f"l'analista esclude '{key}' ma il calcolo lo rileva: vale il calcolo"))
    return out
