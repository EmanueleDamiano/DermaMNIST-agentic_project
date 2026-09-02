"""Composizione dell'agente.

Fase 1: un agente singolo con `create_agent` (LangChain 1.x). Quello che restituisce e' un
grafo LangGraph compilato, quindi puo' diventare un nodo di uno StateGraph piu' grande senza
riscritture: e' la strada verso il multi-agente, che non serve ancora aprire.

"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from .llm import build_llm
from .tools import TOOLS

# Prompt volutamente corto e imperativo: con un modello locale ogni frase in piu' e'
# un'occasione per distrarsi dal compito.
SYSTEM_PROMPT = """Sei l'agente che supervisiona la pipeline di classificazione di DermaMNIST,
un dataset di immagini dermatoscopiche con 7 classi di lesioni cutanee.

Regole di lavoro:
- Usa i tool per ottenere qualsiasi numero. Non inventare mai metriche, conteggi o run_id.
- Se non conosci ancora il dataset, chiama inspect_dataset prima di proporre una configurazione.
- Dopo ogni training chiama get_run_report per interpretare il risultato.
- L'accuracy da sola e' fuorviante: il dataset e' sbilanciato 58 a 1. Ragiona su balanced
  accuracy, macro-F1, AUC e recall sulla classe mel (melanoma), che e' quella a maggior
  costo clinico di errore.
- Per classificare un'immagine locale usa classify_image con il percorso del file. Se l'utente
  indica una cartella, usa prima list_images_in_directory per vedere cosa contiene.
- Riporta sempre la confidenza insieme alla classe predetta, e ricorda che il modello e'
  addestrato su immagini 28x28: la predizione e' indicativa, non diagnostica.
- Se la richiesta e' ambigua, o un risultato e' inatteso, chiama ask_human invece di indovinare.
- Rispondi in italiano, conciso, riportando i numeri che hai davvero ottenuto dai tool."""

# I tool che richiedono approvazione umana prima di essere eseguiti (FR-G4).
# train_model e' l'unico che consuma minuti di calcolo: e' il punto giusto dove mettere
# un freno mentre si mette a punto il sistema.
APPROVAL_REQUIRED = {"train_model": True}


def build_agent(
    model=None,
    checkpointer=None,
    require_approval: bool = True,
    tools=None,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Costruisce l'agente.

    Args:
        model: istanza di chat model. Se None usa il modello locale via Ollama.
            Passare un `ScriptedChatModel` permette di testare il grafo senza LLM.
        checkpointer: necessario perche' gli interrupt possano sospendere e riprendere.
            Di default `InMemorySaver` (la sessione vive quanto il processo).
        require_approval: se True, `train_model` richiede approvazione umana.
        tools: superficie di tool alternativa, per i test.
    """
    middleware = []
    if require_approval:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on=APPROVAL_REQUIRED,
                description_prefix="L'agente chiede di eseguire",
            )
        )

    return create_agent(
        model=model if model is not None else build_llm(),
        tools=tools if tools is not None else TOOLS,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
