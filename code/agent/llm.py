"""Binding del modello linguistico.

Tutto cio' che dipende dal provider vive in questo file. Il grafo, i tool e il registro di
audit non sanno quale modello stia girando: sostituire Ollama con un'API cloud e' un cambio
di questa sola funzione.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

# Modello di default: piccolo e veloce, adatto al ciclo di sviluppo.
# Per la validazione conviene un modello piu' capace (es. qwen3.6), impostando
# la variabile d'ambiente DERMA_AGENT_MODEL.
DEFAULT_MODEL = "qwen3.5:4b"
ENV_VAR = "DERMA_AGENT_MODEL"


def resolve_model_name(model: str | None = None) -> str:
    return model or os.environ.get(ENV_VAR) or DEFAULT_MODEL


def build_llm(model: str | None = None, temperature: float = 0.0, num_ctx: int = 8192):
    """Costruisce il client del modello locale.

    temperature=0 di default: l'agente deve scegliere tool, non essere creativo. Con un
    modello piccolo la temperatura alta si traduce quasi sempre in tool call malformate.
    """
    return ChatOllama(
        model=resolve_model_name(model),
        temperature=temperature,
        num_ctx=num_ctx,
        validate_model_on_init=True,
    )


def check_ollama(model: str | None = None) -> tuple[bool, str]:
    """Verifica che il server Ollama risponda e che il modello richiesto sia presente.

    Restituisce (ok, messaggio). Pensato per essere chiamato prima di avviare l'agente:
    e' molto meglio fallire subito con una diagnosi leggibile che a meta' conversazione.
    """
    name = resolve_model_name(model)
    try:
        import ollama
    except ImportError:
        return False, "Pacchetto 'ollama' non installato: pip install langchain-ollama"

    try:
        listed = ollama.list()
    except Exception as exc:
        return False, (
            f"Server Ollama non raggiungibile ({type(exc).__name__}: {exc}). "
            "Avvialo con `ollama serve` oppure apri l'app Ollama."
        )

    available = [m.model for m in listed.models]
    if name in available:
        return True, f"Ollama attivo, modello '{name}' disponibile."
    # Ollama accetta 'qwen3.5:4b' anche se listato come 'qwen3.5:4b'; tollera l'assenza del tag.
    if any(a.split(":")[0] == name.split(":")[0] for a in available):
        return True, f"Ollama attivo, modello '{name}' risolvibile. Disponibili: {', '.join(available)}"
    return False, (
        f"Modello '{name}' non trovato in Ollama. Disponibili: {', '.join(available) or 'nessuno'}.\n"
        f"Scaricalo con `ollama pull {name}` oppure imposta {ENV_VAR} su uno di quelli elencati."
    )
