# AgenticDerma

AgenticDerma trains, evaluates, and explains a seven-class classifier over the DermaMNIST benchmark, then makes that pipeline available through a chat interface backed by a language model. Every number the system reports, whether a probability, a validation score, or a learning rate, comes from a Python function that ran before any text was generated, and the language model's role is limited to describing that evidence, retrieving supporting literature, and narrating the handoffs between agents. This separation is what lets a small, locally hosted model sit in the loop without becoming the source of the system's numerical claims. If the language model is unavailable, training and prediction continue exactly as before, because neither depends on it.

Four agents divide the deterministic work. Agent1 plans the hyperparameters each training candidate starts from, either by reading the previous completed run for that architecture from the experiment registry and scaling the configured base learning rate from its macro-F1, or by validating a set of values supplied directly by the user. Agent2 executes training against that plan, adjusting the learning rate after every epoch when the plan was automatic, and holding it fixed when the values were manual. Agent3 compares the three finished candidates on validation macro-F1, falling back to balanced accuracy, macro-AUROC, and accuracy in that order when scores tie, then freezes the selected network and later applies it to the test split and to any uploaded image. Agent4 turns a raw prediction into a result a person can read: it computes an input-gradient attribution, attaches the cited class description for the predicted label, and confirms that the prediction, the attribution, and the explanation all reference the same trace ID before either is shown.

A fifth role, AgenticDerma itself, sits outside this chain. It reads the user's message or image, decides which of the four agents the request calls for, maintains the two-tier memory described below, and writes the sentences a person reads, including the ones that describe what another agent just did. None of this generated language can change a number: probabilities, metrics, and saved weights are fixed before AgenticDerma writes about them, and a response that makes a clinical claim without citing one of the retrieved passages is rejected and rewritten before it reaches the interface.

## Problem

A DermaMNIST model can report a strong test score while still being affected by duplicate images, split leakage, class imbalance, or weak minority-class performance, and a generated explanation carries a separate risk when it is not tied to classifier evidence or to an external source. AgenticDerma treats both failure modes as engineering problems rather than as caveats to disclose after the fact. The full workflow, from data audit through training, selection, testing, and explanation, runs under explicit control, with every claim traceable to the function that produced it.

## State of the art

MedMNIST v2 standardizes biomedical classification benchmarks and defines DermaMNIST as a 10,015-image, seven-class dermoscopy set at 28 by 28 RGB resolution [1]. That compact format is what makes repeated experiments practical on ordinary hardware, but compactness says nothing about correctness, and the benchmark's small size makes any duplicate or leaked image a larger fraction of the reported score than it would be in a larger dataset. Abhishek, Jain, and Hamarneh audited DermaMNIST and its source dataset, HAM10000, and found duplicate images, cross-split leakage, and labeling problems that measurably changed performance once corrected for [2]. Li et al. approached the same weakness from a documentation angle, proposing a dermatology dataset nutrition label that records provenance, known limitations, and risk before a dataset is used for training [9]. Between them, these two lines of work turn dataset identity and split integrity from an assumption into evidence that has to be produced, which is why AgenticDerma's training pipeline keeps a dataset checksum, an audit of exact and near-duplicate images, and a record of which classes appear in which split, all written before a model sees the data.

Once the data itself is trustworthy, the harder question is what counts as a good classifier. Dakhli and Barhoumi built saliency information directly into the training loss and evaluated classification jointly with the resulting explanation on HAM10000 and PH2 [5], while Zuo, Wang, and Wang fused clinical images, dermoscopy, and patient metadata through adaptive weighting and used uncertainty to control how much each modality contributed to the final decision [6]. DermaMNIST supplies neither multiple modalities nor patient metadata, so that fusion architecture does not transfer directly, but the underlying lesson does: a single aggregate accuracy figure hides exactly the minority-class failures that matter most in a seven-class, imbalanced dataset. AgenticDerma selects its final model on macro-F1 rather than accuracy, with balanced accuracy and macro-averaged one-vs-rest AUROC as tie-breakers, so that a candidate cannot win by being right about the common classes and wrong about the rare ones.

The gap between a benchmark result and a clinical claim widens further once systems operate at scale. Yan et al. trained PanDerm, a foundation model, on more than two million images spanning multiple institutions and imaging modalities and reported gains across a broad range of dermatology tasks [3]. Nahm et al. reviewed dermatology AI applications that have reached clinical approval and found that deployment depends on validation and workflow fit at least as much as on model accuracy [7]. Han et al. evaluated a skin-disease system against hospital and population-scale usage data and showed how prevalence shifts and out-of-distribution inputs change real-world performance in ways a held-out test set cannot capture [8]. Prospective evidence remains comparatively thin: Laiouar-Pedari et al. pooled eleven prospective studies and found melanoma diagnostic performance comparable between AI and dermatologists on average, alongside substantial heterogeneity and risk of bias across the studies [12], while Anriot et al. found that expert dermatologists still outperformed current AI models in a realistic multiclass setting [13]. Set next to a foundation model trained on two million images or a system validated against national health records, a 10,015-image, 28 by 28 benchmark project has no basis for a clinical claim, and AgenticDerma does not make one: probabilities are reported as a distribution over seven benchmark classes, attached to citations that describe the class, never to a claim that the classifier examined a patient.

Explainability is subject to the same discipline. Chanda et al. studied 76 dermatologists using eye tracking and found that a dermatologist-oriented explainability interface improved balanced melanoma diagnostic accuracy over standard AI support [4], evidence that an explanation is a measurable system output and not a visual add-on whose value can be assumed. At 28 by 28 resolution, fine spatial detail is not available to attribute, so AgenticDerma treats its input-gradient attribution as coarse evidence of what the classifier responded to, and keeps it separate in the output from the cited medical fact retrieved for the predicted class. The two describe different things, and merging them into one explanation would let one imply support for the other.

The coordination problem sits one layer above the model. Agent-based medical AI is moving from single prompts toward systems that call tools and divide work across specialized roles, and the evidence on that shift is mixed enough to be instructive. Ferber et al. connected an autonomous oncology agent to specialist models and retrieval tools and found that the orchestration improved task performance over the base language model alone [10]. Chen et al. reported that a multi-agent conversational framework improved diagnostic reasoning over single-model baselines on rare-disease cases [11]. Two scoping reviews, one by Collaco et al. and one by Yu et al., found that healthcare agents remain largely prototypes, with safety practice and real-world validation still uneven across the field [14, 15]. AgentClinic's benchmarking reached a related conclusion from measurement rather than survey: tool-using clinical agents make larger errors precisely when a task requires sequential actions and interaction with external tools [16]. Read together, these results argue against giving a language model open-ended control over a pipeline's numerical outputs, and for the opposite design: narrow, fixed tool interfaces, and a complete log of every call, at exactly the points where a probability, a metric, or an acceptance decision gets produced. That is the boundary AgenticDerma's four agents and its coordinator are built around, and it is enforced in code rather than by instruction. The language model cannot call a training, evaluation, or scoring function, and every agent's tool calls are checked against an explicit allow list before they run.

Individually, data integrity, class-balanced evaluation, scoped explainability, and tool-bounded agent coordination are each well studied. Combining all four in one reproducible pipeline, where each control is enforced by an interface rather than left to convention, is what the literature above leaves open at the system level for a benchmark this size, and it is the gap AgenticDerma is built to close.

## Objectives and research questions

The project was scoped around five objectives: verify the data basis before training on it, build a classifier by comparing a small set of model families under one protocol, structure the workflow as agents with defined tools and permission boundaries, verify prediction and explanation through fixed metrics and grounded retrieval, and close the loop with an independent review that the acceptance evidence can be checked against. Three research questions follow from those objectives, and each is answered by a specific part of the running system rather than by argument.

Can specialized agents coordinate DermaMNIST experiments while preserving reproducibility, test isolation, and a complete action log? Agent1 through Agent4 answer this by construction. Each has a fixed tool list enforced at the interface level, the final test split is unavailable to Agent1 and Agent2, and every tool call is written to the project registry with its status before the next step runs, whether that call succeeds, fails, or is rejected as unauthorized.

Which classification strategy is strongest when selection uses class-balanced validation metrics and leakage-aware data checks? Agent3 compares ResNet-18, EfficientNet-B0, and ConvNeXt-Tiny under identical preprocessing and a fixed seed policy, selects on validation macro-F1 with balanced accuracy and macro-AUROC as tie-breakers, and freezes the winner before it ever sees the test split. The dataset audit that precedes training records duplicate and cross-split findings as a leakage-aware sensitivity check rather than folding them silently into the primary result.

Can the system add useful lesion information and visual evidence without letting generated text alter the classifier decision? The prediction, its probabilities, and the attribution are all fixed before AgenticDerma generates a word of explanation, and a generated response that makes a clinical claim without an exact citation label from the retrieved passages is rejected and rewritten before it reaches the interface. The classifier's output and the retrieved dermatology literature are kept as two separate kinds of evidence throughout the system, never merged into a single unverified claim.

## Install and run

Python 3.11 or newer is required. From the project folder:

```powershell
python run.py
```

The first run installs `requirements.txt` and builds the knowledge index; later runs skip both once they exist. The launcher then presents three choices.

**Full Process** trains all three candidates, selects and tests one, and analyzes an image with it. Choose `1`, give an image path or press Enter for the bundled sample, then set a target validation accuracy and an epoch limit when prompted; training stops early on whichever is reached first. A further prompt asks whether Agent1 should plan the hyperparameters automatically or use values entered directly, and pressing Enter keeps automatic. Results land in `output/full_process/`, and the selected weights replace `model/model.pt`. The same run is reachable directly:

```powershell
python main/derma_agent.py full input/sample_derma.png --target-accuracy 0.70 --max-epochs 3 --yes
python main/derma_agent.py full input/sample_derma.png --learning-rate 0.002 --weight-decay 0.0001 --batch-size 128 --yes
```

Supplying `--learning-rate` is what switches Agent1 from automatic planning to manual: batch size and weight decay default from the project configuration when left out, but the learning rate itself is required to enter manual mode. See Hyperparameter control below for what changes between the two.

**Final Output** applies the saved model to one image without retraining. Choose `2`; the launcher refuses if `model/model.pt` is missing or its checksum no longer matches the manifest, and points to Full Process instead. Output goes to `output/final_output/`.

```powershell
python main/derma_agent.py test input/test_samples/05_melanoma.png
```

**Platform** starts the chat interface at `http://127.0.0.1:8000` and opens it in a browser.

```powershell
python platform/app.py
```

Attach an image and send it to run Final Output inline. Write "train automatically" to start Full Process, which opens the training plan modal so the target accuracy, epoch limit, and hyperparameter mode are confirmed before anything runs. The **Process** button opens a live overlay showing which agent is active, the handoffs between them, and the current stage, without leaving the conversation; the **+** button starts a new session with its own memory.

Two commands support verification rather than daily use: `python main/derma_agent.py verify` checks every project file, checksum, and deliverable for consistency, and `python main/derma_agent.py notebook` rebuilds and re-executes the reproducible dataset audit notebook against the current data.

## Workflows

### Image analysis

The platform accepts PNG, JPG, WebP, and BMP images. Agent3 applies the same preprocessing used during evaluation and returns all seven class probabilities, then Agent4 creates the attribution and attaches the cited description for the leading class. The result card shows the predicted class, its confidence, whether the uncertainty rule was triggered, the attribution image, and the measured test accuracy of the saved model, plus the cited class fact with a link to its source, so the label is never presented without the reference it came from.

### Model training

Agent1 plans, and Agent2 trains ResNet-18, EfficientNet-B0, and ConvNeXt-Tiny in sequence against whatever plan Agent1 produced, stopping a candidate early either when validation accuracy reaches the requested target or when validation macro-F1 stops improving for two consecutive epochs. Agent3 then compares the three finished runs and freezes the winner in `model/model.pt`.

#### Hyperparameter control

Agent1 plans each candidate's starting learning rate, weight decay, and batch size in one of two ways, chosen once per run. **Automatic** is the default: Agent1 reads the last completed run for that model architecture from the experiment registry and scales the configured base learning rate up if that run's macro-F1 was strong, down if it was weak, then Agent2 keeps adjusting the rate after every epoch based on whether loss fell and validation macro-F1 improved. **Manual** takes the three values from the user, validates them against the same bounds Agent1 enforces automatically, and turns off the per-epoch adjustment for that run, so the rate given is the rate every epoch trains at, with no exceptions. The platform's training modal exposes this as a single dropdown next to the target accuracy and epoch limit; the CLI takes the same values through `--learning-rate`, `--weight-decay`, and `--batch-size`.

### Technical discussion

AgenticDerma can discuss the current model, its latest output, training decisions, uncertainty, and dermoscopy concepts. Retrieval runs locally over nine indexed NCBI sources spanning all seven DermaMNIST classes: general dermoscopy chapters, the international consensus papers on dermoscopic criteria and terminology, the HAM10000 and MedMNIST v2 dataset papers, and dedicated chapters on seborrheic keratosis, cherry hemangioma, and dysplastic nevi. A question about vascular lesions or benign keratoses is therefore answered from indexed material rather than from the melanoma-heavy coverage a general dermoscopy source alone would give. Each clinical sentence carries the label of the passage it came from; a response that omits one is rewritten once with that requirement restated, and if it still lacks a label the reply is replaced by the retrieved passage itself, so a citation is never fabricated to satisfy the check. Model probabilities stay separate from this clinical evidence throughout: they describe the classifier's distribution over seven benchmark classes, not a claim the retrieved literature is asked to support.

### Process inspection

The **Process** button opens a live overlay without leaving the conversation. Its graph is drawn as an actual node-link diagram: AgenticDerma sits as the coordinating hub, with an edge to each of the four agents, and a second row of edges marking the order they execute in; whichever edge and node are currently active are highlighted as the job moves through them. The adjacent timeline records the same handoffs as short, specific sentences rather than raw status strings. Training progress, learning-rate decisions, memory state, and knowledge-index status update while a job runs. Closing the overlay returns to the same chat and does not interrupt the job.

System readiness is checked automatically in the background on load and every fifteen seconds after, but nothing about that check is shown until asked. The top bar carries one compact status pill, a colored dot and a short word, that only turns amber with an explicit label when something needs attention, such as no trained model yet. Clicking the pill opens a small panel with the full breakdown of classifier, dataset, knowledge index, and model connection; it stays closed otherwise, so the interface does not narrate its own health checks.

## Agents

| Component | Responsibility |
|---|---|
| AgenticDerma | Reads the request, selects the workflow, maintains session memory, retrieves literature, and writes every sentence shown to the user. |
| Agent1 | Plans or validates each candidate's starting learning rate, weight decay, and batch size, and fixes the tuning mode for the run. |
| Agent2 | Trains the three candidates against Agent1's plan, adjusting the learning rate per epoch only in automatic mode, and records every run. |
| Agent3 | Selects a candidate by validation macro-F1, freezes it, runs the final test, and classifies uploaded images with the saved model. |
| Agent4 | Produces the attribution, attaches cited class information, and checks that prediction, attribution, and explanation share one trace ID. |

Every agent's tool calls are restricted to an explicit allow list and logged before and after execution; a call outside that list is rejected and recorded as a permission error rather than silently skipped. The language model cannot execute any of these tools or modify a probability, metric, saved weight, or training limit: those values exist before AgenticDerma writes a single word about them.

## Language model

Training and prediction do not require a language model. The platform offers five connections for discussion and handoff text:

- **Local Ollama** keeps prompts on the machine and needs no account. Install [Ollama](https://ollama.com/download), then run `ollama pull qwen3:1.7b` once.
- **Groq** runs open models on Groq's hosted inference. Create a key at [console.groq.com](https://console.groq.com), open **Model connection**, select Groq, and paste the key.
- **Google Gemini** uses Gemini's OpenAI-compatible endpoint. Create a key at [aistudio.google.com](https://aistudio.google.com), select Google Gemini, and paste the key.
- **OpenRouter** uses the `openrouter/free` router by default. Create an OpenRouter key, open **Model connection**, select OpenRouter, and paste the key.
- **OpenAI-compatible API** accepts an HTTPS chat-completions endpoint, model name, and key for any other provider.

The connection form tests the model before applying the change, so a bad endpoint or key is caught immediately rather than surfacing mid-conversation. API keys stay in server process memory and are cleared when the platform stops; none are written to disk. The same settings can be supplied before launch through `AGENTICDERMA_LLM_PROVIDER`, `AGENTICDERMA_LLM_URL`, `AGENTICDERMA_LLM_MODEL`, and `AGENTICDERMA_API_KEY`.

## Memory and retrieval

Deterministic memory stores model readiness, the latest prediction, the latest training summary, and agent states in `output/memory/deterministic.json`. Event memory records user messages, actions, and handoffs in `output/memory/events.jsonl`. When a response needs context, AgenticDerma ranks eligible events by relevance, recency, and session, with a recorded seed for the small stochastic term that keeps repeated questions from always surfacing the identical slice of history. Generated chat text is excluded from this pool unless it passed the grounding path, which stops an earlier ungrounded answer from becoming evidence for a later one.

The knowledge index holds the nine sources described above as BM25-ranked passages, each carrying a stable label and its title and link. When the language model's answer is invalid, missing a citation or containing language the safety check rejects, the rewritten or fallback answer draws directly from the top-ranked passage for that query rather than from a fixed template, so even a failed generation still returns material relevant to what was asked.

## Outputs

| Location | Contents |
|---|---|
| `model/model.pt` | Selected trained classifier. |
| `output/final_output/` | Prediction and attribution from Final Output. |
| `output/full_process/` | Training, selection, evaluation, prediction, and review records. |
| `output/platform/sessions/` | Files produced by individual platform jobs. |
| `output/memory/` | Current state and chronological events. |
| `main/work_packages/` | Structured deliverables produced by the implementation, one folder per stage of the pipeline. |

## Structure

```text
AgenticDerma/
├── run.py
├── input/
├── model/
├── output/
├── platform/
│   └── app.py
├── main/
│   ├── agentic_derma.py
│   ├── derma_agent.py
│   ├── knowledge.py
│   ├── knowledge_base/
│   ├── project_config.json
│   └── work_packages/
└── tests/
    └── test_system.py
```

`run.py` is the entry point. `main/derma_agent.py` holds the model workflow: data, training, evaluation, attribution, and the four agents that execute it. `main/agentic_derma.py` holds routing, memory, retrieval integration, and language-model access, which is the coordinator and the safety checks around generated text. `platform/app.py` serves the responsive interface and the local API that connects the two.

## Test

```powershell
python -m unittest discover -s tests -v
```

The suite covers metrics, model shapes, automatic and manual training controls, agent boundaries, citation grounding, knowledge retrieval across all seven classes, launcher behavior, and interface requirements.

## References

[1] J. Yang, R. Shi, D. Wei, et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification," Scientific Data, vol. 10, art. 41, 2023.

[2] K. Abhishek, A. Jain, and G. Hamarneh, "Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets," Scientific Data, vol. 12, art. 196, 2025.

[3] S. Yan, Z. Yu, C. Primiero, et al., "A multimodal vision foundation model for clinical dermatology," Nature Medicine, vol. 31, pp. 2691 to 2702, 2025.

[4] T. Chanda, S. Haggenmueller, T. C. Bucher, et al., "Dermatologist-like explainable AI enhances melanoma diagnosis accuracy: eye-tracking study," Nature Communications, vol. 16, art. 4739, 2025.

[5] R. Dakhli and W. Barhoumi, "Improving skin lesion classification through saliency-guided loss functions," Computers in Biology and Medicine, vol. 192, pt. B, art. 110299, 2025.

[6] L. Zuo, Z. Wang, and Y. Wang, "A multi-stage multi-modal learning algorithm with adaptive multimodal fusion for improving multi-label skin lesion classification," Artificial Intelligence in Medicine, vol. 162, art. 103091, 2025.

[7] W. J. Nahm, N. Sohail, J. Burshtein, M. Goldust, and M. Tsoukas, "Artificial Intelligence in Dermatology: A Comprehensive Review of Approved Applications, Clinical Implementation, and Future Directions," International Journal of Dermatology, vol. 64, pp. 1568 to 1583, 2025.

[8] S. S. Han, S. I. Cho, G. Fabian, et al., "Planet-wide performance of a skin disease AI algorithm validated in Korea," npj Digital Medicine, vol. 8, art. 603, 2025.

[9] Y. Li, M. Taylor, K. S. Chmielinski, et al., "Improving dataset transparency in dermatologic Artificial Intelligence using a dataset nutrition label," npj Digital Medicine, vol. 8, art. 641, 2025.

[10] D. Ferber, O. S. M. El Nahhas, G. Woelflein, et al., "Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology," Nature Cancer, vol. 6, pp. 1337 to 1349, 2025.

[11] X. Chen, H. Yi, M. You, et al., "Enhancing diagnostic capability with multi-agents conversational large language models," npj Digital Medicine, vol. 8, art. 159, 2025.

[12] S. Laiouar-Pedari, A. Kuehn, C. Wies, et al., "Prospective Evidence on Artificial Intelligence-Assisted Melanoma Diagnostics: A Systematic Review and Meta-Analysis," JAMA Dermatology, vol. 162, no. 5, pp. 478 to 487, 2026.

[13] J. Anriot, S. Yan, C. Coste, et al., "Limits of Artificial Intelligence Models for Skin Cancer Diagnosis in Realistic Settings," JAMA Dermatology, vol. 162, no. 7, pp. 701 to 708, 2026.

[14] B. G. Collaco, S. A. Haider, S. Prabha, et al., "The role of agentic artificial intelligence in healthcare: a scoping review," npj Digital Medicine, vol. 9, art. 345, 2026.

[15] K. Yu, S. Zhou, Y. Hou, et al., "Multimodal artificial intelligence agents in healthcare: a scoping review," npj Digital Medicine, 2026.

[16] S. Schmidgall, R. Ziaei, C. Harris, et al., "AgentClinic: a multimodal benchmark for tool-using clinical AI agents," npj Digital Medicine, vol. 9, art. 499, 2026.
