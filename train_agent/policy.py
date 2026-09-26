"""
Deterministic fallback for the two LLM roles: the agent still works, and is
testable, without any LLM. It is deliberately simple and conservative - it is
the "logical agent" baseline the LLM is meant to improve on:

    first run      the plan's architecture (FPViT by default), AdamW, effective-
                   number class weights, the 'default' augmentation preset
    plateau        warm restart from the best run with lr x 0.3
    still improving (budget ran out)  warm restart with the same lr
    overfitting    warm restart with weight decay x 3
    collapsed classes with class_weight 'effective'  new run with 'inverse'
    2 segments in a row without a real gain  stop
"""

from __future__ import annotations

from train_agent.space import DEFAULT_LR

AUG_RATIONALE_DEFAULT = (
    "Preset 'default': gruppo diedrale completo (flip orizzontale e verticale + rotazioni di 90°), "
    "esattamente label-preserving perché le lesioni dermoscopiche non hanno un orientamento "
    "canonico; RandomResizedCrop 0.8–1.0 per traslazione e scala senza bordi neri (che imiterebbero "
    "la vignettatura del dermatoscopio); color jitter leggero con hue 0.02, perché il colore è un "
    "segnale diagnostico. Nessun cutout: è un regolarizzatore e non c'è ancora overfitting misurato."
)
MIN_GAIN = 0.005


def propose(plan: dict, trials: list[dict]) -> dict:
    ok = [t for t in trials if t["summary"]["status"] != "error" and t["summary"].get("checkpoint")]
    if not ok:
        arch = plan["arch"]
        return {
            "action": "new_run", "arch": arch,
            "hparams": {"optimizer": "adamw", "lr": DEFAULT_LR.get(arch, 3e-4), "weight_decay": 0.05,
                        "epochs": plan["max_epochs_per_run"], "class_weight": "effective"},
            "augmentation": {"preset": "default", "overrides": {}},
            "augmentation_rationale": AUG_RATIONALE_DEFAULT,
            "rationale": f"Primo run: {arch} con AdamW e pesi per classe 'effective' (Cui et al.), "
                         "la leva più morbida contro lo sbilanciamento 67% nevi.",
            "addresses": [],
        }

    gains = [t["facts"].get("improvement_vs_campaign_best") for t in trials[-2:]]
    if len(trials) >= 3 and all(g is not None and g < MIN_GAIN for g in gains):
        return {"action": "stop", "rationale": "Due segmenti consecutivi senza un guadagno reale "
                                               f"(soglia {MIN_GAIN}): altri tentativi simili non aiutano."}

    best = max(ok, key=lambda t: t["summary"]["best_score"] or -1)
    last = trials[-1]
    flags = set(last["facts"].get("flags", []))
    base = best["proposal"]
    hp = dict(base["hparams"])
    common = {"arch": base["arch"], "augmentation": base["augmentation"],
              "augmentation_rationale": "Augmentation invariata rispetto al run di partenza: "
                                        "la diagnosi non indica un problema di generalizzazione "
                                        "che l'augmentation possa correggere."}

    if "overfitting" in flags:
        hp["weight_decay"] = min(0.5, hp["weight_decay"] * 3)
        return {"action": "warm_restart", "parent_run": best["summary"]["run"], "hparams": hp, **common,
                "rationale": f"Overfitting misurato (gap {last['facts']['gap_at_best']}→"
                             f"{last['facts']['gap_at_end']}): riparto dal miglior checkpoint con "
                             "weight decay ×3.", "addresses": ["overfitting"]}
    if last["facts"].get("collapsed_classes") and hp.get("class_weight") == "effective":
        hp["class_weight"] = "inverse"
        return {"action": "new_run", "hparams": hp, **common,
                "rationale": "Classi ancora a recall 0 ("
                             + ", ".join(last["facts"]["collapsed_classes"])
                             + "): passo ai pesi 'inverse', la correzione più forte.",
                "addresses": ["collapsed_classes"]}
    if "plateau" in flags:
        hp["lr"] = hp["lr"] * 0.3
        return {"action": "warm_restart", "parent_run": best["summary"]["run"], "hparams": hp, **common,
                "rationale": f"Plateau: nessun miglioramento per {plan['patience']} epoche. Riparto dal "
                             "miglior checkpoint con lr ×0.3 e optimizer nuovo.", "addresses": ["plateau"]}
    return {"action": "warm_restart", "parent_run": last["summary"]["run"], "hparams": hp, **common,
            "rationale": "Il segmento si è fermato per budget mentre migliorava ancora: continuo dallo "
                         "stesso checkpoint con lo stesso lr.", "addresses": ["still_improving"]}


def analyse(facts: dict) -> dict:
    flags = facts.get("flags", [])
    parts = [f"Stop: {facts.get('trigger')}. Miglior {facts.get('selection_metric')} "
             f"{facts.get('score')} all'epoca {facts.get('best_epoch')}/{facts.get('epochs_completed')}."]
    if "overfitting" in flags:
        parts.append(f"Il gap train/val cresce ({facts['gap_at_best']}→{facts['gap_at_end']}).")
    if facts.get("collapsed_classes"):
        parts.append("Classi a recall 0: " + ", ".join(facts["collapsed_classes"]) + ".")
    rec = ("ridurre l'lr e ripartire dal checkpoint" if "plateau" in flags else
           "aumentare la regolarizzazione" if "overfitting" in flags else
           "continuare l'addestramento")
    return {"verdict": " ".join(parts), "overfitting": "overfitting" in flags,
            "underfitting": "underfitting" in flags, "plateau": "plateau" in flags,
            "likely_causes": [], "recommendation": rec, "knowledge_used": []}
