"""Every location the system reads or writes, in one place.

    input/      what goes in: sample images, images uploaded on the platform
    output/     what comes out: prediction and review logs, training runs, test reports
    model/      trained models: the shipped baseline and the promoted ones
    database/   data: the DermaMNIST dataset and the two knowledge bases
    system/     the code and its configuration
"""

from pathlib import Path

SYSTEM = Path(__file__).resolve().parent
ROOT = SYSTEM.parent

INPUT = ROOT / "input"
SAMPLES = INPUT / "samples"
UPLOADS = INPUT / "uploads"

OUTPUT = ROOT / "output"
PREDICTIONS_LOG = OUTPUT / "predictions.jsonl"
REVIEWS_LOG = OUTPUT / "reviews.jsonl"
CAMPAIGNS = OUTPUT / "campaigns"          # training campaigns of the Training agent
RUNS = OUTPUT / "runs"                    # manual train.py runs
EVALUATIONS = OUTPUT / "evaluations"      # isolated test reports

MODEL = ROOT / "model"
BASELINE = MODEL / "baseline"
PROMOTED = MODEL / "promoted"
EXCLUSIONS = MODEL / "ensemble_exclusions.json"   # runs left out of the ensemble, and why

DATABASE = ROOT / "database"
DATASET = DATABASE / "dermamnist"         # dermamnist.npz and DermaMNIST-C/E, downloaded on first use
CLINICAL_KB = DATABASE / "clinical_kb.json"
TRAINING_KB = DATABASE / "training_kb.json"

CONFIG = SYSTEM / "config"
