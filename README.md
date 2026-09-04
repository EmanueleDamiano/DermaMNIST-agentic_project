# DermaAgent

DermaAgent classifies dermoscopic images into the seven DermaMNIST classes. It trains and compares three image classifiers, tests the selected model, creates an attribution image, adds cited class information, and records the result.

Research and education only.

## Goal and questions

The goal is a reproducible image-classification workflow with clear agent responsibilities, data checks, model comparison, final testing, explanation, and review.

- Can three agents complete the workflow with consistent records and defined responsibilities?
- Which of the three classifiers gives the strongest class-balanced result?
- Can the system add attribution and cited class information without changing the classifier probabilities?

The project covers the DermaMNIST benchmark. It does not provide clinical diagnosis, treatment advice, or hospital deployment.

## Run

Install the packages once:

```powershell
python -m pip install -r requirements.txt
```

```powershell
python run.py
```

Choose one number:

1. Run the complete process.
2. Test the trained model.
3. Open the visual platform.

The launcher checks required files and packages. Choice 2 also checks the trained-model file and checksum. Choice 1 downloads DermaMNIST when the dataset is missing.

The visual platform has two processing choices: **Test trained model** and **Run full process**. The full-process form includes target accuracy and maximum epochs.

## System flow

1. Audit the official DermaMNIST file and splits.
2. Prepare the 28 × 28 RGB images.
3. Agent1 trains ResNet-18, EfficientNet-B0, and ConvNeXt-Tiny.
4. Agent2 compares the models and saves the selected model.
5. Agent2 runs the final test on all 2,005 test images.
6. Agent3 creates attribution and cited class information.
7. Agent3 saves the result and checks the project records.

## Agents

| Agent | Work |
|---|---|
| Agent1 | Trains the three classifier candidates and saves their records. |
| Agent2 | Evaluates the candidates, selects one model, and runs the final test. |
| Agent3 | Creates attribution, class information, output, and review records. |

Each agent has a fixed action list. Actions are recorded in `main/project_registry.jsonl`.

## Work packages

| Work package | Purpose | Deliverables |
|---|---|---|
| WP01 — Research and Requirements | Defines requirements, methods, roles, and acceptance rules. | D1.1–D1.3 |
| WP02 — Data | Registers DermaMNIST, checks data quality, and defines preprocessing. | D2.1–D2.3 |
| WP03 — Training | Defines the training protocol and trains three classifier families. | D3.1–D3.3 |
| WP04 — Evaluation | Compares the models and saves the selected model. | D4.1–D4.3 |
| WP05 — Enrichment and XAI | Creates attribution and cited class information. | D5.1–D5.3 |
| WP06 — Testing | Calculates final metrics, robustness results, and failure cases. | D6.1–D6.3 |
| WP07 — Output | Builds the prediction report and prototype output. | D7.1–D7.3 |
| WP08 — Review | Checks the required records and produces the acceptance report. | D8.1–D8.2 |

The deliverables are in `main/work_packages/WP01` through `main/work_packages/WP08`.

## Data, models, and results

DermaMNIST contains 10,015 RGB images at 28 × 28 pixels:

1. Actinic keratoses and intraepithelial carcinoma
2. Basal cell carcinoma
3. Benign keratosis-like lesions
4. Dermatofibroma
5. Melanoma
6. Melanocytic nevi
7. Vascular lesions

The three models use the same normalization, augmentation, class-weighted loss, seed, and metrics. The report includes accuracy, macro-F1, balanced accuracy, macro-AUROC, per-class metrics, confusion matrix, calibration error, Brier score, negative log-likelihood, and robustness checks.

The included complete run selected `efficientnet_b0_28`. Test accuracy is 0.5397, macro-F1 is 0.3641, balanced accuracy is 0.4502, and macro-AUROC is 0.8669. All eight acceptance checks passed.

## Folder structure

| Folder or file | Contents |
|---|---|
| `run.py` | Opens the three-choice run menu and checks each option before starting. |
| `input/` | Included real DermaMNIST image and user input images. |
| `main/` | Complete workflow, configuration, dataset, work packages, and deliverables. |
| `model/` | Selected trained model used by Final Output and the platform. |
| `output/final_output/` | Saved-model prediction and attribution. |
| `output/full_process/` | Training, selection, testing, review, prediction, and plots. |
| `output/platform/` | Platform results separated into `final_output` and `full_process`. |
| `platform/app.py` | Interactive platform with saved-model and full-process choices. |
| `Project_Overview.xlsx` | Project objectives, work packages, deliverables, agents, methods, risks, and acceptance rules. |
| `tests/test_system.py` | Automated system tests. |
| `RUN_GUIDE.md` | Exact commands and expected outputs. |

## Test

```powershell
python -m unittest discover -s tests -v
```

See `RUN_GUIDE.md` for the shortest path through each option.
