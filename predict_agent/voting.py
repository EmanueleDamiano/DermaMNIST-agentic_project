"""
Ensemble voting over the local models, per image. Pure Python, deterministic.

Three views of the same predictions are computed, because on this dataset they
disagree in informative ways:

  soft vote       probabilities averaged with weight = model skill (balanced
                  accuracy above chance). The primary decision.
  hard vote       each model votes its argmax class, weighted by its own val
                  *precision for that class* - "when this model says melanoma,
                  how often is it right?". A model that spams one class gets
                  little credit for it.
  most confident  the single model with the highest top probability. Reported,
                  never decisive on its own: a 1-epoch model collapsed on class
                  5 is very confident and very wrong.

The agreement between the three, and the margin between the first two classes
of the soft vote, set the confidence level the reasoning step starts from.
"""

from __future__ import annotations

from predict_agent.models import CLASS_NAMES, NUM_CLASSES, ModelCard

# Soft-vote margin (top1 - top2 probability) below which the call is a coin flip.
LOW_MARGIN = 0.10
HIGH_MARGIN = 0.25
# ...and 'high' also needs the vote itself to be a majority of the probability
# mass: three weak models collapsed on the same class agree easily.
HIGH_MIN_PROB = 0.50
# A class enters the candidate set if its soft-vote probability reaches this.
CANDIDATE_MIN_PROB = 0.15


def _argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


def vote_one(per_model: dict[str, list[float]], cards: dict[str, ModelCard]) -> dict:
    """Combine one image's per-model probabilities into a decision plus evidence."""
    if not per_model:
        raise ValueError("no model produced a prediction")

    # --- per-model opinions ------------------------------------------------
    opinions = []
    for name, probs in per_model.items():
        card = cards[name]
        top = _argmax(probs)
        opinions.append({
            "model": name,
            "predicted_class_id": top,
            "predicted_class": CLASS_NAMES[top],
            "confidence": round(probs[top], 4),
            "skill_weight": round(card.skill, 4),
            "val_precision_for_this_class": round(card.val_per_class_precision[top], 4),
            "blind_classes": card.blind_classes,
        })

    # --- soft vote: skill-weighted mean of probabilities -------------------
    total_w = sum(cards[n].skill for n in per_model)
    soft = [sum(cards[n].skill * p[c] for n, p in per_model.items()) / total_w
            for c in range(NUM_CLASSES)]
    ranked = sorted(range(NUM_CLASSES), key=lambda c: soft[c], reverse=True)
    soft_top, soft_second = ranked[0], ranked[1]
    margin = soft[soft_top] - soft[soft_second]

    # --- hard vote: argmax votes weighted by per-class val precision -------
    hard = [0.0] * NUM_CLASSES
    for op in opinions:
        # epsilon: a class a model has never predicted correctly still counts
        # as one (tiny) voice, so the tally shows who voted for what.
        hard[op["predicted_class_id"]] += max(op["val_precision_for_this_class"], 0.01)
    hard_top = _argmax(hard)
    n_agree = sum(op["predicted_class_id"] == soft_top for op in opinions)

    # --- most confident single model ---------------------------------------
    most_conf = max(opinions, key=lambda op: op["confidence"])

    # --- confidence level ---------------------------------------------------
    views_agree = soft_top == hard_top == most_conf["predicted_class_id"]
    if views_agree and margin >= HIGH_MARGIN and soft[soft_top] >= HIGH_MIN_PROB:
        level = "high"
    elif soft_top != hard_top or margin < LOW_MARGIN:
        level = "low"
    else:
        level = "medium"

    # Models that could not have voted for the winner: their dissent is not evidence.
    blind_to_winner = [op["model"] for op in opinions if soft_top in op["blind_classes"]]

    candidates = sorted(
        {c for c in range(NUM_CLASSES) if soft[c] >= CANDIDATE_MIN_PROB}
        | {soft_top, soft_second, hard_top}
        | {op["predicted_class_id"] for op in opinions},
        key=lambda c: soft[c], reverse=True,
    )

    return {
        "vote_class_id": soft_top,
        "vote_class": CLASS_NAMES[soft_top],
        "vote_probability": round(soft[soft_top], 4),
        "runner_up_class_id": soft_second,
        "runner_up_class": CLASS_NAMES[soft_second],
        "margin": round(margin, 4),
        "confidence_level": level,
        "soft_vote": {CLASS_NAMES[c]: round(soft[c], 4) for c in ranked},
        "hard_vote": {CLASS_NAMES[c]: round(hard[c], 4)
                      for c in sorted(range(NUM_CLASSES), key=lambda c: hard[c], reverse=True)
                      if hard[c] > 0},
        "hard_vote_class_id": hard_top,
        "models_agreeing_with_vote": f"{n_agree}/{len(opinions)}",
        "most_confident_model": most_conf,
        "models_blind_to_vote_class": blind_to_winner,
        "candidate_class_ids": candidates,
        "per_model": opinions,
    }
