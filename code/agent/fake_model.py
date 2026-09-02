"""Modello finto e deterministico, per testare il grafo senza LLM.

E' il pezzo che rende il sistema debuggabile. Con un modello locale i fallimenti sono di due
tipi — il grafo e' sbagliato, oppure il modello non ha chiamato i tool giusti — e senza un
modo di eliminare la seconda causa si finisce a modificare i prompt sperando che qualcosa
cambi.

`ScriptedChatModel` restituisce una sequenza fissata di risposte: se il test passa con lui e
fallisce con Ollama, il problema e' del modello, non dell'architettura.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Riproduce una lista di AIMessage, uno per invocazione.

    Esauriti i messaggi, ripete l'ultimo: evita loop infiniti se il grafo chiama il modello
    piu' volte del previsto, e rende il fallimento visibile invece che bloccante.
    """

    responses: list[AIMessage] = []
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        """Accetta i tool e li ignora: le risposte sono gia' decise dallo script."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        message = self.responses[index] if self.responses else AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call_message(tool_name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Costruisce un AIMessage che richiede una tool call, come farebbe un vero modello."""
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final_message(text: str) -> AIMessage:
    """Costruisce la risposta finale, senza tool call: chiude il loop dell'agente."""
    return AIMessage(content=text)
