"""Language-model providers shared by every agent, the CLIs and the platform.

    from llm import build_chat_model, structured, resolve_model

A model is addressed as "<provider id>:<model id>", for example
"anthropic:claude-opus-5-5", "ollama:qwen3.6" or "openrouter:qwen/qwen3-235b-a22b".
Providers are declared in system/config/llm.json; see llm/registry.py.
"""

from llm.registry import (DEFAULT_ANTHROPIC_MODEL, ProviderError, available_models, build_chat_model,
                          default_model, delete_provider, fetch_models, get_provider, load_config,
                          provider_status, providers_overview, resolve_model, save_provider,
                          set_default_model, set_secret, split_spec, structured, test_model)

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL", "ProviderError", "available_models", "build_chat_model", "default_model",
    "delete_provider", "fetch_models", "get_provider", "load_config", "provider_status",
    "providers_overview", "resolve_model", "save_provider", "set_default_model", "set_secret",
    "split_spec", "structured", "test_model",
]
