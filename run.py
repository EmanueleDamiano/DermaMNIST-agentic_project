"""Interactive DermaAgent launcher."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_FILE = ROOT / "main" / "derma_agent.py"
PLATFORM_FILE = ROOT / "platform" / "app.py"
CONFIG_FILE = ROOT / "main" / "project_config.json"
DATA_FILE = ROOT / "main" / "data" / "dermamnist.npz"
MANIFEST_FILE = ROOT / "main" / "work_packages" / "WP04" / "D4.3_frozen_model_manifest.json"
MODEL_DIR = ROOT / "model"
DEFAULT_IMAGE = ROOT / "input" / "sample_derma.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_issue(*required: Path) -> str | None:
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    return f"Missing project file: {', '.join(missing)}" if missing else None


def dependency_issue() -> str | None:
    check = subprocess.run(
        [sys.executable, "-c", "import numpy, torch, matplotlib, PIL, openpyxl, nbformat, nbclient"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return None
    return "Required packages are missing. Run: python -m pip install -r requirements.txt"


def trained_model_issue() -> str | None:
    if not MANIFEST_FILE.is_file():
        return "No trained model was found. Choose 1 to run the complete process first."
    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        model_path = (ROOT / manifest["model_file"]).resolve()
        if model_path.parent != MODEL_DIR.resolve() or not model_path.is_file():
            return "The trained model file is missing. Choose 1 to train a new model."
        if file_sha256(model_path) != manifest["model_sha256"]:
            return "The trained model checksum is invalid. Choose 1 to train a new model."
    except (KeyError, OSError, json.JSONDecodeError):
        return "The trained model record is invalid. Choose 1 to train a new model."
    return None


def choose_image() -> Path | None:
    while True:
        shown = DEFAULT_IMAGE.relative_to(ROOT)
        entered = input(f"Image path [{shown}] or Q to cancel: ").strip().strip('"')
        if entered.lower() == "q":
            return None
        path = DEFAULT_IMAGE if not entered else Path(entered)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            print(f"Image not found: {path}")
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            print("Use a JPG, JPEG, PNG, BMP, or WebP image.")
            continue
        return path


def run_command(arguments: list[str]) -> int:
    try:
        return subprocess.run([sys.executable, *arguments], cwd=ROOT).returncode
    except OSError as exc:
        print(f"Could not start the command: {exc}")
        return 2


def run_complete_process() -> int:
    issue = project_issue(MAIN_FILE, CONFIG_FILE)
    if issue:
        print(issue)
        return 2
    issue = dependency_issue()
    if issue:
        print(issue)
        return 2
    image = choose_image()
    if image is None:
        print("Cancelled.")
        return 0
    if not DATA_FILE.is_file():
        print("DermaMNIST is not present and will be downloaded automatically.")
    return run_command([str(MAIN_FILE), "full", str(image)])


def test_trained_model() -> int:
    issue = project_issue(MAIN_FILE, CONFIG_FILE)
    if issue:
        print(issue)
        return 2
    issue = dependency_issue() or trained_model_issue()
    if issue:
        print(issue)
        return 2
    image = choose_image()
    if image is None:
        print("Cancelled.")
        return 0
    return run_command([str(MAIN_FILE), "test", str(image)])


def wait_for_platform(process: subprocess.Popen[bytes]) -> bool:
    for _ in range(30):
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000", timeout=1) as response:
                return response.status == 200 and b"DermaAgent" in response.read()
        except OSError:
            time.sleep(0.2)
    return False


def start_platform() -> int:
    issue = project_issue(PLATFORM_FILE, MAIN_FILE, CONFIG_FILE)
    if issue:
        print(issue)
        return 2
    issue = dependency_issue()
    if issue:
        print(issue)
        return 2
    try:
        process = subprocess.Popen([sys.executable, str(PLATFORM_FILE)], cwd=ROOT)
    except OSError as exc:
        print(f"Could not start the platform: {exc}")
        return 2
    try:
        if not wait_for_platform(process):
            print("The platform could not start. Check whether port 8000 is already in use.")
            return 2
        print("Platform: http://127.0.0.1:8000")
        print("Press Ctrl+C to stop it.")
        webbrowser.open("http://127.0.0.1:8000")
        return process.wait()
    except KeyboardInterrupt:
        print("\nPlatform stopped.")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def print_menu() -> None:
    data_status = "READY" if DATA_FILE.is_file() else "DOWNLOAD REQUIRED"
    model_status = "READY" if trained_model_issue() is None else "NOT READY"
    input_status = "READY" if DEFAULT_IMAGE.is_file() else "SELECT FILE"
    print("\nDERMA AGENT")
    print("=" * 58)
    print(f"DATASET  {data_status:<18} MODEL  {model_status:<10} INPUT  {input_status}")
    print("-" * 58)
    print("1  FULL PROCESS       Train, test, and create final output")
    print("2  FINAL OUTPUT       Test the trained model")
    print("3  PLATFORM           Open the visual interface")
    print("=" * 58)


def main() -> int:
    print_menu()
    while True:
        try:
            choice = input("Choose 1, 2, or 3: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0
        if choice in {"1", "2", "3"}:
            break
        print("Enter 1, 2, or 3.")
    try:
        return {"1": run_complete_process, "2": test_trained_model, "3": start_platform}[choice]()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
