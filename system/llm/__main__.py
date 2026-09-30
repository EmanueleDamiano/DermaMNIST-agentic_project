"""
Check the language-model providers from the terminal.

    python run.py llm                          list providers, their status and models
    python run.py llm test anthropic:claude-opus-5-5
    python run.py llm default ollama:qwen3.6   set the default model ("" to clear)
"""

from __future__ import annotations

import sys

from llm import providers_overview, set_default_model, test_model


def main(argv: list[str]) -> int:
    if argv[:1] == ["test"] and len(argv) == 2:
        r = test_model(argv[1])
        print(f"{argv[1]}: " + (f"OK in {r['seconds']} s, reply: {r['reply']!r}" if r["ok"] else f"FAILED: {r['error']}"))
        return 0 if r["ok"] else 1
    if argv[:1] == ["default"] and len(argv) == 2:
        set_default_model(argv[1])
        print(f"Default model set to: {argv[1] or '(automatic)'}")
        return 0
    if argv:
        print(__doc__)
        return 2
    o = providers_overview()
    for p in o["providers"]:
        mark = "ready  " if p["ready"] else "not set"
        print(f"[{mark}] {p['id']:<12} {p['label']:<22} {p['detail']}")
        if p["models"]:
            print(" " * 11 + "models: " + ", ".join(p["models"][:12]) + (" ..." if len(p["models"]) > 12 else ""))
    print(f"\nDefault model: {o['default_model'] or '(automatic)'} -> used now: {o['resolved_default'] or 'none, agents run without an LLM'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
