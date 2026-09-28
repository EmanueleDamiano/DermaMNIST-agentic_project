# DermaAgent — Use Cases

Scope: research and education on the DermaMNIST benchmark (28×28, seven classes). No use case covers clinical diagnosis, treatment advice, or real patient data.

## Actors

| Actor | Description |
| --- | --- |
| **Researcher** | Primary user. Runs predictions, launches training, inspects traces. |
| **Human reviewer** | Answers `interrupt()`s: clarifications, flagged predictions, training plans, proposals, promotions. Can be the researcher. |
| **Orchestrator** | Routes each request to the right agent; surfaces interrupts. |
| **Testing agent** | Runs the local model ensemble, votes, reasons over memory and knowledge base. |
| **Reviewer agent** | Audits the tester's trace with independent retrieval and deterministic checks. |
| **Training agent** | Plans, proposes, runs and diagnoses training; requests promotion. |
| **LLM (optional)** | Ollama, Anthropic or OpenRouter. Replaced by deterministic rules with `--no-llm`. |

## Summary

| ID | Use case | Primary actor | Agents involved | Human in the loop |
| --- | --- | --- | --- | --- |
| UC1 | Classify a single image | Researcher | Orchestrator, Testing, Reviewer | Only if issues and `--ask-human` |
| UC2 | Classify a batch with human review of flagged cases | Researcher | Orchestrator, Testing, Reviewer | Yes, on flagged images |
| UC3 | Run without an LLM | Researcher | Orchestrator, Testing, Reviewer | Optional |
| UC4 | Launch a training campaign from a natural-language request | Researcher | Orchestrator, Training | Plan always; runs per autonomy level |
| UC5 | Promote a new model into the ensemble | Training agent | Training, Human reviewer | Always |
| UC6 | Clarify an ambiguous request | Orchestrator | Orchestrator, Human reviewer | Always |
| UC7 | Detect a contradiction with a past prediction | Testing agent | Testing, Reviewer | If flagged |
| UC8 | Audit a decision after the fact | Researcher | Web platform, logs | No |

---

## UC1 — Classify a single image

**Goal.** Obtain a class, the ensemble evidence, and a reviewed explanation for one dermoscopy image.

**Preconditions.** At least one checkpoint under `runs/`, `runs_agent/`, `runs test solo training pyramid/` or `models_promoted/`.

**Trigger.** `python -m orchestrator test_samples/05_melanoma.png`, or image upload in the web platform.

**Main flow.**
1. The orchestrator sees an image and routes to the testing agent deterministically, with no LLM call.
2. The testing agent runs every local model and computes a skill-weighted soft vote, a precision-weighted hard vote, and the most confident model.
3. It looks up past executions of the same image (deterministic memory, SHA-256) and retrieves knowledge-base chunks (stochastic memory, TF-IDF).
4. The LLM reasons over this evidence and returns the final class and rationale.
5. The reviewer agent runs independent retrieval and the deterministic checks, and identifies the decisive model.
6. No warning or critical finding: the reviewer summarises the reasoning for the user.

**Alternative flows.**
- 5a. A warning or critical finding appears (e.g. melanoma/nevus margin < 0.20, claims about the image): the image is marked **REQUIRES HUMAN REVIEW** → UC2.
- 4a. The LLM fails or proposes an override outside the candidate set: the override is rejected in code and logged.

**Postconditions.** One line in `runs_predict/predictions_log.jsonl` and one in `runs_predict/reviews_log.jsonl`, linked by `execution_id`.

---

## UC2 — Classify a batch with human review of flagged cases

**Goal.** Process many images and stop only on those that need a person.

**Trigger.** `python -m orchestrator <folder> --ask-human`

**Main flow.**
1. Steps 1–5 of UC1 for each image.
2. For every image with a warning or critical finding, the graph stops with an `interrupt()`.
3. The human sees the checks, the decisive model and the rationale, and answers *accept*, *reject* or *accept with note*.
4. The decision is written to the review log.

**Alternative flows.**
- 3a. No human at the keyboard (stdin closed): the decision is recorded as **rejected**. Acceptance is never given silently.

**Postconditions.** Every flagged prediction carries an explicit human decision.

---

## UC3 — Run without an LLM

**Goal.** Reproducible run with no API key and no LLM variance.

**Trigger.** `python -m orchestrator test_samples --no-llm`, `python -m train_agent ... --no-llm`

**Main flow.**
1. Prediction: the ensemble vote is the final decision.
2. Review: only deterministic checks and template summaries.
3. Training: `policy.py` replaces the LLM proposer.

**Postconditions.** Same logs as UC1/UC4. Output depends only on checkpoints and code.

---

## UC4 — Launch a training campaign from a natural-language request

**Goal.** Improve the ensemble on a stated objective within a budget.

**Trigger.** `python -m orchestrator --train "improve dermatofibroma recall, 30 minutes"`, or the training panel in the web platform.

**Main flow.**
1. The orchestrator LLM extracts intent `train` and the declared constraints only (architecture, runs, minutes, autonomy).
2. The training agent writes a campaign plan. **The human always approves the plan.**
3. The agent proposes a run: architecture, hyperparameters, class weighting, augmentation, each justified.
4. The proposal is validated (max 3 retries) and checked by the training reviewer.
5. The autonomy gate decides whether the proposal needs a human:
   - `supervised`: every run needs approval.
   - `guarded` (default): only small lr / weight-decay / epoch changes on the same architecture pass automatically.
   - `autonomous`: everything within the budget.
6. `train.py` runs in a new folder; the agent analyses the curves and decides: new run, warm restart, or stop.
7. On stop → UC5.

**Alternative flows.**
- 4a. The training reviewer raises a warning or critical finding: the proposal goes to the human regardless of autonomy level.
- 5a. The human asks for a different proposal → back to step 3; or stops → step 7.

**Postconditions.** Every run in its own folder with config, curves and validation metrics. No checkpoint is overwritten.

---

## UC5 — Promote a new model into the ensemble

**Goal.** Add a trained model to prediction only if it helps.

**Preconditions.** A completed candidate run from UC4.

**Main flow.**
1. The agent computes ensemble balanced accuracy on validation, with and without the candidate. The test split is never loaded.
2. `improves` is true only if the gain is ≥ `MIN_GAIN`.
3. **The human always approves or rejects the promotion.**
4. On approval, the checkpoint is **copied** under a new name into `models_promoted/`.

**Postconditions.** From the next run, the testing agent loads the promoted model (UC1).

---

## UC6 — Clarify an ambiguous request

**Goal.** Avoid acting on a guessed intent.

**Trigger.** Text request with intent `unclear` (e.g. "hi, how are you?"), or no LLM available to parse it.

**Main flow.**
1. The orchestrator raises an `interrupt` of kind `clarify_intent`.
2. The human chooses prediction, training, or answer.
3. The orchestrator resumes on the chosen branch (UC1, UC4).

---

## UC7 — Detect a contradiction with a past prediction

**Goal.** Surface when the same image now gets a different answer.

**Trigger.** An image whose SHA-256 is already in the execution log.

**Main flow.**
1. The testing agent retrieves past executions by exact hash.
2. If the new class differs, the reviewer raises a warning, usually because a new checkpoint entered the ensemble.
3. The image follows UC2 if `--ask-human` is set.

---

## UC8 — Audit a decision after the fact

**Goal.** Reconstruct why a prediction or promotion was made.

**Trigger.** `python -m webapp`, or reading the JSONL logs.

**Main flow.**
1. The researcher opens the execution in the platform: agent graph, communication trace, votes, checks.
2. The context window shows the full LLM input; the knowledge-base window shows the retrieved chunks.
3. For training: per-epoch curves, proposals, human decisions.

**Postconditions.** Every number shown comes from code; every LLM statement can be traced to its inputs.

---

## Out of scope

- Clinical diagnosis or triage of real patients.
- Images outside the DermaMNIST distribution (resized inputs are flagged, not rejected).
- Any use of the test split by the agents.
