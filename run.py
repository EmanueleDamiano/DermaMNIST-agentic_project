#!/usr/bin/env python3
"""
DermaAgent: the single entry point.

    python run.py                     start the platform and open it in the browser
    python run.py --port 8080         another port
    python run.py --no-browser        do not open the browser

Everything is done from the platform. The same tools are also available here,
for scripting or for a machine without a browser:

    python run.py orchestrator input/samples --no-llm     classify and review images
    python run.py tester input/samples --no-llm           Testing agent alone
    python run.py trainer --arch resnet18 --no-llm        a training campaign
    python run.py train --arch resnet18 --epochs 5        one manual training run
    python run.py predict --image input/samples --checkpoint model/baseline/baseline_paper/best_model.pt
    python run.py evaluate --checkpoint model/baseline/baseline_paper/best_model.pt
    python run.py llm                                     language-model providers
    python run.py promote output/runs_224/<run>            promotion gate for a model trained outside the agent
    python run.py build-kb                                rebuild database/training_kb.json
"""

from __future__ import annotations

import os
import runpy
import sys
import threading
import warnings
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = ROOT / "system"

# tool name -> ("module", name) or ("script", file in system/)
TOOLS = {
    "orchestrator": ("module", "orchestrator"),
    "tester": ("module", "tester.agent"),
    "trainer": ("module", "trainer"),
    "train": ("script", "train.py"),
    "predict": ("script", "predict.py"),
    "evaluate": ("script", "evaluate.py"),
    "llm": ("module", "llm"),
    "promote": ("module", "trainer.promotion"),
    "build-kb": ("module", "trainer.build_kb"),
}


def check_setup() -> bool:
    if sys.version_info < (3, 10):
        print(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}).")
        return False
    missing = []
    for module, package in (("torch", "torch"), ("medmnist", "medmnist"), ("sklearn", "scikit-learn"),
                            ("langgraph", "langgraph"), ("langchain_core", "langchain-core"), ("PIL", "pillow")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing packages: " + ", ".join(missing))
        print("Install them with:  pip install -r requirements.txt")
        return False
    return True


def run_tool(name: str, args: list[str]) -> None:
    kind, target = TOOLS[name]
    sys.argv = [f"run.py {name}"] + args
    if kind == "module":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(str(SYSTEM / target), run_name="__main__")


def run_platform(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="python run.py", description="Start the DermaAgent platform",
                                epilog="Tools: " + ", ".join(TOOLS) + ". Use: python run.py <tool> --help")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    a = p.parse_args(args)
    from ui.app import main
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    main(a.host, a.port)


def entry() -> int:
    sys.path.insert(0, str(SYSTEM))
    os.chdir(ROOT)                       # relative paths such as input/samples work from here
    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package")
    if not check_setup():
        return 1
    args = sys.argv[1:]
    if args and args[0] in TOOLS:
        run_tool(args[0], args[1:])
        return 0
    if args and not args[0].startswith("-"):
        print(f"Unknown tool '{args[0]}'. Available: {', '.join(TOOLS)}.")
        return 2
    run_platform(args)
    return 0


if __name__ == "__main__":
    sys.exit(entry())
