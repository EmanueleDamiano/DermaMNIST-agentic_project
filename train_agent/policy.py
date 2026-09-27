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
    "Preset 'default': full dihedral group (horizontal and vertical flips + 90° rotations), "
    "exactly label-preserving because dermoscopic lesions have no canonical orientation; "
    "RandomResizedCrop 0.8–1.0 for translation and scale without black borders (which would mimic "
    "dermoscope vignetting); light colour jitter with hue 0.02, because colour is a diagnostic "
    "signal. No cutout: it is a regulariser and there is no measured overfitting yet."
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
            "rationale": f"First run: {arch} with AdamW and 'effective' class weights (Cui et al.), "
                         "the gentlest lever against the 67% nevi imbalance.",
            "addresses": [],
        }

    gains = [t["facts"].get("improvement_vs_campaign_best") for t in trials[-2:]]
    if len(trials) >= 3 and all(g is not None and g < MIN_GAIN for g in gains):
        return {"action": "stop", "rationale": "Two consecutive segments without a real gain "
                                               f"(threshold {MIN_GAIN}): more similar attempts will not help."}

    best = max(ok, key=lambda t: t["summary"]["best_score"] or -1)
    last = trials[-1]
    flags = set(last["facts"].get("flags", []))
    base = best["proposal"]
    hp = dict(base["hparams"])
    common = {"arch": base["arch"], "augmentation": base["augmentation"],
              "augmentation_rationale": "Augmentation unchanged from the starting run: "
                                        "the diagnosis shows no generalisation problem "
                                        "that augmentation could fix."}

    if "overfitting" in flags:
        hp["weight_decay"] = min(0.5, hp["weight_decay"] * 3)
        return {"action": "warm_restart", "parent_run": best["summary"]["run"], "hparams": hp, **common,
                "rationale": f"Measured overfitting (gap {last['facts']['gap_at_best']}→"
                             f"{last['facts']['gap_at_end']}): restarting from the best checkpoint with "
                             "weight decay ×3.", "addresses": ["overfitting"]}
    if last["facts"].get("collapsed_classes") and hp.get("class_weight") == "effective":
        hp["class_weight"] = "inverse"
        return {"action": "new_run", "hparams": hp, **common,
                "rationale": "Classes still at recall 0 ("
                             + ", ".join(last["facts"]["collapsed_classes"])
                             + "): switching to 'inverse' weights, the strongest correction.",
                "addresses": ["collapsed_classes"]}
    if "plateau" in flags:
        hp["lr"] = hp["lr"] * 0.3
        return {"action": "warm_restart", "parent_run": best["summary"]["run"], "hparams": hp, **common,
                "rationale": f"Plateau: no improvement for {plan['patience']} epochs. Restarting from the "
                             "best checkpoint with lr ×0.3 and a fresh optimizer.", "addresses": ["plateau"]}
    return {"action": "warm_restart", "parent_run": last["summary"]["run"], "hparams": hp, **common,
            "rationale": "The segment stopped on budget while still improving: continuing from the "
                         "same checkpoint with the same lr.", "addresses": ["still_improving"]}


def analyse(facts: dict) -> dict:
    flags = facts.get("flags", [])
    parts = [f"Stop: {facts.get('trigger')}. Best {facts.get('selection_metric')} "
             f"{facts.get('score')} at epoch {facts.get('best_epoch')}/{facts.get('epochs_completed')}."]
    if "overfitting" in flags:
        parts.append(f"The train/val gap grows ({facts['gap_at_best']}→{facts['gap_at_end']}).")
    if facts.get("collapsed_classes"):
        parts.append("Classes at recall 0: " + ", ".join(facts["collapsed_classes"]) + ".")
    rec = ("lower the lr and restart from the checkpoint" if "plateau" in flags else
           "increase regularisation" if "overfitting" in flags else
           "keep training")
    return {"verdict": " ".join(parts), "overfitting": "overfitting" in flags,
            "underfitting": "underfitting" in flags, "plateau": "plateau" in flags,
            "likely_causes": [], "recommendation": rec, "knowledge_used": []}
