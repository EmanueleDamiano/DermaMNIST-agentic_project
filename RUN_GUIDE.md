# Run Guide

Run each command from the `DermaAgent` folder.

## Install

```powershell
python -m pip install -r requirements.txt
```

No API key is required.

## Start

```powershell
python run.py
```

The menu shows:

1. Run the complete process.
2. Test the trained model.
3. Open the visual platform.

Enter the number and follow the displayed question. Press Enter at the image question to use `input/sample_derma.png`.

## Choice 1 — run the complete process

Choice 1 checks the project files and Python packages. If DermaMNIST is missing, it downloads the dataset. It then asks for the image, target validation accuracy, maximum epochs, and confirmation.

Each model stops when it reaches the target, stops improving, or reaches the maximum epochs. The workflow uses all official training, validation, and test images.

The selected trained model is saved as `model/model.pt`.

Output:

- `output/full_process/run_summary.json`
- `output/full_process/training_results.json`
- `output/full_process/model_selection.json`
- `output/full_process/final_test_results.json`
- `output/full_process/review.json`
- `output/full_process/prediction.json`
- `output/full_process/attribution.png`
- `output/full_process/confusion_matrix.png`
- `output/full_process/calibration_plot.png`

## Choice 2 — test the trained model

Choice 2 checks the model record, model file, and checksum. If the model is unavailable or invalid, it tells you to run choice 1. It then checks the image before running the model.

Choice 2 uses `model/model.pt`.

Output:

- `output/final_output/prediction.json`
- `output/final_output/attribution.png`

## Choice 3 — visual platform

Choice 3 starts the platform and opens `http://127.0.0.1:8000`.

The page contains two processing choices:

- **Test trained model** checks one uploaded image with the saved model.
- **Run full process** trains the models, selects one, runs the final test, checks the uploaded image, and reviews the results.

The full-process choice asks for target accuracy and maximum epochs on the page. The workflow section shows the active stage and completion progress.

Saved-model results go to `output/platform/final_output/`. Complete-process results go to `output/platform/full_process/`.

Stop the platform with `Ctrl+C`.

## Direct commands

```powershell
python main/derma_agent.py test input/sample_derma.png
python main/derma_agent.py full input/sample_derma.png --target-accuracy 0.70 --max-epochs 3 --yes
python platform/app.py
```

## Automated tests

```powershell
python -m unittest discover -s tests -v
```

The tests check metrics, model output shapes, data selection, agent limits, trace consistency, probability preservation, and training stop settings.
