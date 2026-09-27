"""
The two LLM roles of the training agent, as structured-output calls.

    proposer  given the plan, the campaign so far, the computed facts of the last
              segment, the deterministic memory and knowledge-base passages:
              what to do next (new run / warm restart / stop) and why.
    analyst   given one segment's computed facts and curve, plus KB passages:
              what happened. It never sees the action space, so it cannot
              reason backwards from a change it already has in mind.

Both only *propose*: space.validate_proposal, review.py and space.gate decide
what actually runs and who has to approve it.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from fpvit.augment import AUG_SEARCH_SPACE, PRESETS, _coerce
from fpvit.zoo import ARCHITECTURES
from predict_agent.models import CLASS_NAMES

FLAGS = Literal["plateau", "overfitting", "underfitting", "still_improving", "collapsed_classes"]

PROPOSER_SYSTEM = f"""\
You are the training agent of a dermatology research project. You plan and adapt
the training of image classifiers on DermaMNIST (28x28 dermatoscopic images,
7 classes: {json.dumps(dict(enumerate(CLASS_NAMES)))}).

The problem:
- The train split is heavily imbalanced: [228, 359, 769, 80, 779, 4693, 99] per
  class. Class 5 (nevi) is 67 %: accuracy means nothing; the selection metric is
  fixed by the plan (usually balanced accuracy). Per-class recall matters most.
- Validation has ~1000 images, some classes only 10-20: differences below ~0.01
  are noise.
- Available architectures: {json.dumps(ARCHITECTURES)}.
  Unless the plan fixes the architecture, you choose it and say why.

How training runs: each segment is one train.py run. It stops by itself when
the selection metric has not improved for `patience` epochs (a PLATEAU) or when
the segment's time budget is spent. Then you are called again with the facts.

Your options:
- new_run: train from scratch (any architecture, hyper-parameters, augmentation).
- warm_restart: continue from `parent_run`'s best checkpoint with a FRESH
  optimizer and schedule, so a new lr / weight_decay takes effect. Same
  architecture and network shape as the parent. The usual answer to a plateau
  while the model is not overfitting: lower the lr (e.g. x0.3).
- stop: nothing worth trying within the budget, or the evidence says further
  runs will not help.

Rules:
- Change ONE thing at a time unless the last run was a dead end, and say which.
- class_weight ('inverse'/'effective') and balanced_sampler are ALTERNATIVES
  against the imbalance: never both.
- `addresses` lists only flags that appear in the facts you were given.
- AUGMENTATION must suit dermoscopy, and `augmentation_rationale` must explain
  it in terms of these images and of the diagnosis:
    * the dihedral group (h/v flips + 90-degree rotations) is exactly label-
      preserving: lesions have no canonical orientation;
    * colour is diagnostic (pigment separates melanoma, nevi, keratoses): keep
      hue jitter tiny (<= 0.03), saturation modest;
    * never fill borders with black (it mimics dermoscope vignetting); crops
      must stay inside the image; aspect distortion deforms lesion shape;
    * cutout / stronger crops are REGULARISERS: use them against measured
      overfitting, not against underfitting;
  Presets: {json.dumps(sorted(PRESETS))}; fields you may override:
  {json.dumps(sorted(AUG_SEARCH_SPACE))}.
- For new_run and warm_restart, FILL EVERY FIELD you decided (optimizer, lr,
  weight_decay, epochs, class_weight, augmentation_preset, ...): only the fields
  are executed. A choice written only in the rationale does not happen.
- Cite knowledge passages by chunk_id in `knowledge_used` only if you used them.
- The test split does not exist for you.
- If a human left feedback, follow it.
Write rationales in the language of the request (English if none), concretely,
quoting the numbers you were given.
"""

ANALYST_SYSTEM = """\
You review one finished training segment of an image classifier on DermaMNIST
(7 imbalanced classes). You get FACTS computed from its curve - they are
authoritative, do not contradict them - the thinned curve, and knowledge-base
passages. Explain what happened and why, in 2-4 sentences, and give ONE
concrete recommendation. You do not choose hyper-parameters. Cite passages by
chunk_id in `knowledge_used` only if you used them. Write in the language of
the request (English if none).
"""


class Proposal(BaseModel):
    """Flat, and with the decisive fields REQUIRED. Tested on a local qwen: with
    optional or nested fields it wrote its choices ('class_weight inverse',
    'dihedral') only in the prose and left the fields empty, so the defaults ran
    instead. Required fields are part of the JSON schema that constrains decoding
    (Ollama, OpenAI, Anthropic tools), so they cannot be skipped. For 'stop' the
    values are ignored."""
    action: Literal["new_run", "warm_restart", "stop"]
    arch: Literal["fpvit", "resnet18", "efficientnet_b0", "convnext_tiny"]
    parent_run: Optional[str] = Field(None, description="warm_restart only: the run to continue from")
    optimizer: Literal["adamw", "sgd"]
    lr: float
    weight_decay: float
    batch_size: Optional[int] = None
    epochs: int = Field(description="upper bound for the segment (the plateau trigger usually stops earlier)")
    class_weight: Literal["none", "inverse", "effective"]
    cb_beta: Optional[float] = None
    balanced_sampler: Optional[bool] = None
    embed_dim: Optional[int] = Field(None, description="FPViT only")
    depth: Optional[int] = Field(None, description="FPViT only")
    num_heads: Optional[int] = Field(None, description="FPViT only")
    augmentation_preset: Literal[tuple(sorted(PRESETS))]
    augmentation_overrides: str = Field("", description="optional, comma-separated field=value, e.g. 'cj_hue=0.01, cutout=true'")
    augmentation_rationale: str = Field(description="REQUIRED, separate from rationale: why this augmentation "
                                                    "suits dermoscopic images and this diagnosis")
    rationale: str = Field(description="why this action and these hyper-parameters")
    addresses: list[FLAGS] = Field(default_factory=list)
    expected_effect: str = ""
    knowledge_used: list[str] = Field(default_factory=list)


HPARAM_FIELDS = ("optimizer", "lr", "weight_decay", "batch_size", "epochs", "class_weight", "cb_beta",
                 "balanced_sampler", "embed_dim", "depth", "num_heads")


class Diagnosis(BaseModel):
    verdict: str
    overfitting: Optional[bool] = None
    underfitting: Optional[bool] = None
    plateau: Optional[bool] = None
    likely_causes: list[str] = Field(default_factory=list)
    recommendation: str
    knowledge_used: list[str] = Field(default_factory=list)


def proposal_to_dict(p: Proposal) -> dict:
    d = p.model_dump()
    out = {k: d[k] for k in ("action", "arch", "parent_run", "augmentation_rationale", "rationale",
                             "addresses", "expected_effect", "knowledge_used") if d.get(k) not in (None, "")}
    out["action"] = p.action
    out["hparams"] = {k: d[k] for k in HPARAM_FIELDS if d.get(k) is not None}
    if p.augmentation_preset or p.augmentation_overrides:
        overrides = {}
        for item in (p.augmentation_overrides or "").split(","):
            if "=" not in item:
                continue
            field, value = (x.strip() for x in item.split("=", 1))
            try:
                overrides[field] = _coerce(field, value)
            except (KeyError, ValueError, TypeError):
                overrides[field] = value              # validation will name it
        out["augmentation"] = {"preset": p.augmentation_preset or "default", "overrides": overrides}
    return out


class LLMRoles:
    def __init__(self, llm, name: str = ""):
        self.name = name or type(llm).__name__
        self._proposer = llm.with_structured_output(Proposal)
        self._analyst = llm.with_structured_output(Diagnosis)

    def propose(self, payload: dict) -> dict:
        out = self._proposer.invoke([("system", PROPOSER_SYSTEM),
                                     ("human", json.dumps(payload, ensure_ascii=False, indent=1, default=str))])
        return proposal_to_dict(out)

    def analyse(self, payload: dict) -> dict:
        out = self._analyst.invoke([("system", ANALYST_SYSTEM),
                                    ("human", json.dumps(payload, ensure_ascii=False, indent=1, default=str))])
        return out.model_dump()
