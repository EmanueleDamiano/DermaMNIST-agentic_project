"""
Provider registry: which language models the agents can use, and how to build them.

Configuration lives in two files under system/config/:

    llm.json       providers (kind, base URL, suggested models, key variable) and
                   the default model. Safe to commit: it never holds a key.
    secrets.json   API keys saved from the platform. Local only (git-ignored,
                   file mode 600). An environment variable always wins over it.

Three provider kinds cover almost every service:

    anthropic   Claude models through the Anthropic API.
    openai      any OpenAI-compatible endpoint: OpenAI, OpenRouter, Groq, DeepSeek,
                Mistral, Together, Google Gemini (OpenAI endpoint), LM Studio, vLLM...
    ollama      local models served by `ollama serve`.

Adding a provider is one entry in llm.json (or the "Add provider" form of the
platform). Nothing in the agents changes: they receive a LangChain chat model.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import paths

CONFIG_DIR = paths.CONFIG
CONFIG_PATH = CONFIG_DIR / "llm.json"
SECRETS_PATH = CONFIG_DIR / "secrets.json"

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
KINDS = ("anthropic", "openai", "ollama")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MODEL_RE = re.compile(r"^[\w.\-:/@]{1,160}$")

DEFAULT_CONFIG: dict = {
    "default_model": "",
    "providers": [
        {"id": "anthropic", "label": "Anthropic (Claude)", "kind": "anthropic",
         "api_key_env": "ANTHROPIC_API_KEY",
         "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"]},
        {"id": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
         "api_key_env": "OPENAI_API_KEY", "models": []},
        {"id": "openrouter", "label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
         "api_key_env": "OPENROUTER_API_KEY", "models": ["qwen/qwen3-235b-a22b"], "options": {"temperature": 0}},
        {"id": "ollama", "label": "Ollama (local)", "kind": "ollama", "base_url": "http://127.0.0.1:11434",
         "models": []},
    ],
}

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"mtime": None, "config": None}
_OLLAMA_CACHE: dict[str, tuple[float, list[str]]] = {}


class ProviderError(ValueError):
    """A configuration or connection problem, worded for the person using the platform."""


# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if private:
        os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_config() -> dict:
    """The provider configuration, re-read whenever the file changes."""
    with _LOCK:
        if not CONFIG_PATH.exists():
            _write_json(CONFIG_PATH, DEFAULT_CONFIG)
        mtime = CONFIG_PATH.stat().st_mtime
        if _CACHE["mtime"] != mtime:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            data.setdefault("default_model", "")
            data.setdefault("providers", [])
            _CACHE.update(mtime=mtime, config=data)
        return json.loads(json.dumps(_CACHE["config"]))


def _save_config(data: dict) -> None:
    with _LOCK:
        _write_json(CONFIG_PATH, data)
        _CACHE["mtime"] = None


def _secrets() -> dict:
    try:
        return json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def set_secret(provider_id: str, key: str) -> None:
    """Store (or, with an empty key, remove) the API key of a provider."""
    with _LOCK:
        data = _secrets()
        if key:
            data[provider_id] = key.strip()
        else:
            data.pop(provider_id, None)
        _write_json(SECRETS_PATH, data, private=True)


def get_provider(provider_id: str) -> dict:
    for p in load_config()["providers"]:
        if p["id"] == provider_id:
            return p
    known = ", ".join(p["id"] for p in load_config()["providers"])
    raise ProviderError(f"Unknown provider '{provider_id}'. Configured providers: {known}.")


def _api_key(p: dict) -> tuple[str, str]:
    """(key, source) with source 'environment', 'saved' or ''."""
    env = p.get("api_key_env")
    if env and os.environ.get(env):
        return os.environ[env], "environment"
    if p["kind"] == "anthropic" and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return os.environ["ANTHROPIC_AUTH_TOKEN"], "environment"
    saved = _secrets().get(p["id"], "")
    return (saved, "saved") if saved else ("", "")


def _requires_key(p: dict) -> bool:
    if p["kind"] == "ollama":
        return False
    return bool(p.get("requires_key", True))


# ---------------------------------------------------------------------------
# Status and discovery
# ---------------------------------------------------------------------------

def _http_json(url: str, headers: dict | None = None, timeout: float = 6.0) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _ollama_base(p: dict) -> str:
    base = os.environ.get("OLLAMA_HOST") if p["id"] == "ollama" and os.environ.get("OLLAMA_HOST") else \
        p.get("base_url") or "http://127.0.0.1:11434"
    base = base.rstrip("/")
    return base if base.startswith("http") else "http://" + base


def _ollama_models(p: dict, timeout: float = 0.8) -> Optional[list[str]]:
    """Installed models, or None when the server does not answer. Cached for 15 s."""
    base = _ollama_base(p)
    hit = _OLLAMA_CACHE.get(base)
    if hit and time.time() - hit[0] < 15:
        return hit[1]
    try:
        names = [m["name"] for m in _http_json(base + "/api/tags", timeout=timeout).get("models", [])]
    except Exception:
        names = None
    _OLLAMA_CACHE[base] = (time.time(), names)
    return names


def provider_status(p: dict) -> dict:
    """Is the provider usable now, and which models does it offer?"""
    if p["kind"] == "ollama":
        names = _ollama_models(p)
        if names is None:
            return {"ready": False, "models": p.get("models", []), "key_source": "",
                    "detail": f"Ollama is not running at {_ollama_base(p)}. Start it with `ollama serve`."}
        if not names:
            return {"ready": False, "models": [], "key_source": "",
                    "detail": "Ollama is running but has no models. Pull one, e.g. `ollama pull qwen3.6`."}
        return {"ready": True, "models": names, "key_source": "",
                "detail": f"{len(names)} local model{'s' if len(names) != 1 else ''} installed."}
    key, source = _api_key(p)
    models = list(p.get("models", []))
    if _requires_key(p) and not key:
        env = p.get("api_key_env")
        how = f"Set {env} in the environment or save a key here." if env else "Save an API key here."
        return {"ready": False, "models": models, "key_source": "", "detail": "No API key. " + how}
    if not models:
        return {"ready": False, "models": [], "key_source": source,
                "detail": "Key available, but no model listed. Add a model name or load the list from the provider."}
    where = {"environment": "key from the environment", "saved": "key saved on this machine"}.get(source, "no key needed")
    return {"ready": True, "models": models, "key_source": source, "detail": f"Ready ({where})."}


def providers_overview() -> dict:
    """Everything the platform shows about providers. Keys are never included."""
    cfg = load_config()
    out = []
    for p in cfg["providers"]:
        st = provider_status(p)
        out.append({"id": p["id"], "label": p.get("label") or p["id"], "kind": p["kind"],
                    "base_url": p.get("base_url", ""), "api_key_env": p.get("api_key_env", ""),
                    "requires_key": _requires_key(p), "configured_models": p.get("models", []), **st})
    return {"providers": out, "default_model": cfg.get("default_model", ""),
            "resolved_default": default_model() or "", "kinds": list(KINDS)}


def available_models() -> list[str]:
    """"provider:model" for every model of every ready provider."""
    out = []
    for p in load_config()["providers"]:
        st = provider_status(p)
        if st["ready"]:
            out += [f"{p['id']}:{m}" for m in st["models"]]
    return out


def default_model() -> Optional[str]:
    """The configured default if its provider is ready, else the first ready model, else None."""
    cfg = load_config()
    wanted = cfg.get("default_model") or ""
    available = available_models()
    if wanted and wanted in available:
        return wanted
    return available[0] if available else None


def fetch_models(provider_id: str) -> list[str]:
    """Ask the provider which models it serves."""
    p = get_provider(provider_id)
    key, _ = _api_key(p)
    try:
        if p["kind"] == "ollama":
            names = _ollama_models(p, timeout=3.0)
            if names is None:
                raise ProviderError(f"Ollama is not reachable at {_ollama_base(p)}.")
            return names
        if p["kind"] == "anthropic":
            base = (p.get("base_url") or "https://api.anthropic.com").rstrip("/")
            data = _http_json(base + "/v1/models?limit=100",
                              {"x-api-key": key, "anthropic-version": "2023-06-01"})
            return [m["id"] for m in data.get("data", [])]
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = _http_json(p["base_url"].rstrip("/") + "/models", headers)
        return sorted(m["id"] for m in data.get("data", []) if m.get("id"))
    except urllib.error.HTTPError as exc:
        hint = " Check the API key." if exc.code in (401, 403) else ""
        raise ProviderError(f"The provider answered {exc.code} {exc.reason}.{hint}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach the provider: {exc.reason}.") from exc


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def save_provider(data: dict) -> dict:
    """Create or update a provider. `api_key` (optional) goes to secrets.json, never to llm.json."""
    pid = str(data.get("id") or "").strip().lower()
    if not ID_RE.match(pid):
        raise ProviderError("The provider id must be 1-32 characters: lowercase letters, digits, '-' or '_'.")
    kind = data.get("kind")
    if kind not in KINDS:
        raise ProviderError(f"The provider type must be one of: {', '.join(KINDS)}.")
    base_url = str(data.get("base_url") or "").strip()
    if kind == "openai" and not base_url:
        raise ProviderError("An OpenAI-compatible provider needs a base URL, e.g. https://api.groq.com/openai/v1.")
    if base_url and not re.match(r"^https?://", base_url):
        raise ProviderError("The base URL must start with http:// or https://.")
    env = str(data.get("api_key_env") or "").strip()
    if env and not re.match(r"^[A-Z_][A-Z0-9_]{0,63}$", env):
        raise ProviderError("The environment variable name must use capital letters, digits and '_'.")
    models = data.get("models") or []
    if isinstance(models, str):
        models = [m.strip() for m in re.split(r"[,\n]", models)]
    models = [m for m in dict.fromkeys(str(m).strip() for m in models) if m]
    bad = [m for m in models if not MODEL_RE.match(m)]
    if bad:
        raise ProviderError(f"Invalid model name: {bad[0]}")
    entry = {"id": pid, "label": str(data.get("label") or pid).strip()[:60], "kind": kind}
    if base_url:
        entry["base_url"] = base_url.rstrip("/")
    if env:
        entry["api_key_env"] = env
    if kind == "openai" and data.get("requires_key") is False:
        entry["requires_key"] = False
    entry["models"] = models
    cfg = load_config()
    old = next((p for p in cfg["providers"] if p["id"] == pid), None)
    if old and old.get("options"):
        entry["options"] = old["options"]
    cfg["providers"] = [entry if p["id"] == pid else p for p in cfg["providers"]] if old \
        else cfg["providers"] + [entry]
    _save_config(cfg)
    if data.get("api_key"):
        set_secret(pid, str(data["api_key"]))
    return entry


def delete_provider(provider_id: str) -> None:
    cfg = load_config()
    if not any(p["id"] == provider_id for p in cfg["providers"]):
        raise ProviderError(f"Unknown provider '{provider_id}'.")
    cfg["providers"] = [p for p in cfg["providers"] if p["id"] != provider_id]
    if split_spec(cfg.get("default_model") or "x:y")[0] == provider_id:
        cfg["default_model"] = ""
    _save_config(cfg)
    set_secret(provider_id, "")


def set_default_model(spec: str) -> None:
    spec = (spec or "").strip()
    if spec:
        split_spec(spec)
    cfg = load_config()
    cfg["default_model"] = spec
    _save_config(cfg)


# ---------------------------------------------------------------------------
# Building models
# ---------------------------------------------------------------------------

def split_spec(spec: str) -> tuple[str, str]:
    """"provider:model" -> (provider, model). Only the first colon splits, because
    Ollama tags contain colons ("qwen3.5:4b"). A bare "claude-..." means Anthropic."""
    spec = (spec or "").strip()
    head, _, tail = spec.partition(":")
    if not tail:
        if spec.startswith("claude-"):
            return "anthropic", spec
        raise ProviderError(f"'{spec}' is not a model address. Use provider:model, e.g. anthropic:{DEFAULT_ANTHROPIC_MODEL}.")
    return head, tail


def resolve_model(spec: Optional[str]) -> Optional[str]:
    """'auto' -> the default model (or None when nothing is available); 'none'/'' -> None."""
    spec = (spec or "").strip()
    if spec.lower() in ("", "none", "off"):
        return None
    if spec.lower() in ("auto", "default"):
        return default_model()
    split_spec(spec)
    return spec


def build_chat_model(spec: str, **overrides):
    """A LangChain chat model for "provider:model"."""
    pid, model_id = split_spec(spec)
    p = get_provider(pid)
    key, _ = _api_key(p)
    if _requires_key(p) and not key:
        env = p.get("api_key_env")
        raise ProviderError(f"No API key for {p.get('label', pid)}"
                            + (f": set {env} or save a key in the platform." if env else "."))
    opts = dict(p.get("options") or {})
    if p["kind"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Thinking is left to the model's default (adaptive on current Claude models,
        # which cannot switch it off). Structured output uses native JSON schema,
        # see structured(), so no forced tool call is needed.
        kwargs = {"model": model_id, "api_key": key, "max_tokens": 16000, **opts}
        if p.get("base_url"):
            kwargs["base_url"] = p["base_url"]
        return ChatAnthropic(**{**kwargs, **overrides})
    if p["kind"] == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama's default context window is small and it truncates SILENTLY: the
        # vote evidence plus retrieved passages would fall off the front.
        kwargs = {"model": model_id, "base_url": _ollama_base(p), "temperature": 0.0,
                  "num_ctx": 32768, "validate_model_on_init": True, **opts}
        return ChatOllama(**{**kwargs, **overrides})
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ProviderError("OpenAI-compatible providers need the package langchain-openai: "
                            "pip install langchain-openai") from exc
    kwargs = {"model": model_id, "base_url": p["base_url"], "api_key": key or "not-needed", **opts}
    return ChatOpenAI(**{**kwargs, **overrides})


def structured(llm, schema):
    """`llm.with_structured_output(schema)` with the method each provider handles best.

    Anthropic: native JSON-schema output. Current Claude models reject a forced
    tool call, which is what LangChain's default method sends.
    OpenAI-compatible: function calling, the variant most gateways support.
    """
    name = type(llm).__name__
    if name == "ChatAnthropic":
        return llm.with_structured_output(schema, method="json_schema")
    if name == "ChatOpenAI":
        return llm.with_structured_output(schema, method="function_calling")
    return llm.with_structured_output(schema)


def test_model(spec: str) -> dict:
    """One short call, to check that the address, the key and the network work."""
    started = time.time()
    try:
        llm = build_chat_model(spec)
        reply = llm.invoke([("human", "Reply with the single word: ready")])
        text = reply.content if isinstance(reply.content, str) else " ".join(
            b.get("text", "") for b in reply.content if isinstance(b, dict))
        return {"ok": True, "seconds": round(time.time() - started, 1), "reply": text.strip()[:200]}
    except Exception as exc:
        return {"ok": False, "seconds": round(time.time() - started, 1),
                "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
