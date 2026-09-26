"""Entry point interattivo dell'agente.

    python -m agent.run_cli --check                     # verifica Ollama e il modello
    python -m agent.run_cli                             # sessione interattiva
    python -m agent.run_cli --task "quante immagini per classe ci sono?"
    python -m agent.run_cli --task "..." --auto-approve # senza conferme, per gli script

Gestisce due tipi di sospensione del grafo, che si riprendono in modo diverso:
- l'approvazione richiesta dal middleware prima di `train_model`;
- la domanda posta dall'agente stesso con il tool `ask_human`.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from langgraph.types import Command

from .audit import AuditLog, set_audit
from .graph import build_agent
from .llm import build_llm, check_ollama, resolve_model_name


def _last_text(result: dict) -> str:
    """Estrae il testo dell'ultimo messaggio dell'agente."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return content if isinstance(content, str) else str(content)
    return "(nessuna risposta testuale)"


def _describe_action(action: dict) -> str:
    name = action.get("name", "?")
    args = action.get("args", {})
    pretty = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "nessun argomento"
    return f"{name}({pretty})"


def _resume_value(payload, auto_approve: bool, audit: AuditLog):
    """Traduce una sospensione del grafo nella risposta con cui riprenderla."""
    # Caso 1: approvazione di tool call richiesta dal middleware.
    if isinstance(payload, dict) and "action_requests" in payload:
        decisions = []
        for action in payload["action_requests"]:
            description = _describe_action(action)
            if auto_approve:
                print(f"  [auto-approvato] {description}")
                audit.event("approval", action=description, decision="approve", auto=True)
                decisions.append({"type": "approve"})
                continue
            print(f"\n  L'agente vuole eseguire: {description}")
            answer = input("  Approvi? [s/N] ").strip().lower()
            if answer in ("s", "si", "sì", "y", "yes"):
                audit.event("approval", action=description, decision="approve", auto=False)
                decisions.append({"type": "approve"})
            else:
                reason = input("  Motivo del rifiuto (invio per nessuno): ").strip()
                audit.event(
                    "approval", action=description, decision="reject", reason=reason, auto=False
                )
                decisions.append({"type": "reject", "message": reason or None})
        return {"decisions": decisions}

    # Caso 2: domanda posta dall'agente con ask_human.
    if isinstance(payload, dict) and payload.get("type") == "question":
        question = payload.get("question", "(domanda vuota)")
        if auto_approve:
            print(f"\n  L'agente chiede: {question}\n  [risposta automatica: procedi]")
            audit.event("human_answer", question=question, answer="procedi", auto=True)
            return "procedi come ritieni opportuno"
        print(f"\n  L'agente chiede: {question}")
        answer = input("  Risposta: ").strip()
        audit.event("human_answer", question=question, answer=answer, auto=False)
        return answer

    # Caso non previsto: si riprende senza contenuto, ma lo si registra.
    audit.event("interrupt_sconosciuto", payload=str(payload)[:500])
    print(f"\n  Sospensione non riconosciuta: {payload}")
    return None


def run_turn(agent, user_input: str, config: dict, audit: AuditLog, auto_approve: bool) -> str:
    """Esegue un turno completo, risolvendo tutte le sospensioni che incontra."""
    audit.event("user_message", text=user_input)
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)

    # Un turno puo' sospendersi piu' volte (approvazione, poi domanda, poi altra approvazione).
    guard = 0
    while "__interrupt__" in result and result["__interrupt__"]:
        guard += 1
        if guard > 20:
            audit.event("errore", message="troppe sospensioni consecutive, turno interrotto")
            return "Turno interrotto: troppe sospensioni consecutive."
        payload = result["__interrupt__"][0].value
        resume = _resume_value(payload, auto_approve, audit)
        result = agent.invoke(Command(resume=resume), config=config)

    answer = _last_text(result)
    audit.event("agent_answer", text=answer)
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.run_cli",
        description="Agente di supervisione per la pipeline DermaMNIST.",
    )
    parser.add_argument("--task", help="Esegue un singolo compito ed esce.")
    parser.add_argument("--model", help="Modello Ollama da usare (default: qwen3.5:4b).")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approva automaticamente le richieste. Da usare solo negli script di verifica.",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Disattiva del tutto il gate di approvazione umana.",
    )
    parser.add_argument("--check", action="store_true", help="Verifica Ollama ed esce.")
    args = parser.parse_args(argv)

    ok, message = check_ollama(args.model)
    if args.check:
        print(message)
        return 0 if ok else 1
    if not ok:
        print(f"Errore: {message}")
        return 1

    audit = AuditLog()
    set_audit(audit)
    audit.event("session_start", model=resolve_model_name(args.model))

    agent = build_agent(
        model=build_llm(args.model) if args.model else None,
        require_approval=not args.no_approval,
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"Agente pronto (modello: {resolve_model_name(args.model)}).")
    print(f"Registro di audit: {audit.path}")

    if args.task:
        print(f"\n> {args.task}\n")
        print(run_turn(agent, args.task, config, audit, args.auto_approve))
        audit.event("session_end")
        return 0

    print("Scrivi una richiesta, oppure 'esci' per terminare.\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("esci", "exit", "quit"):
            break
        try:
            print(f"\n{run_turn(agent, user_input, config, audit, args.auto_approve)}\n")
        except Exception as exc:  # l'agente non deve far cadere la sessione
            audit.event("errore", message=f"{type(exc).__name__}: {exc}")
            print(f"\nErrore durante il turno: {type(exc).__name__}: {exc}\n")

    audit.event("session_end")
    print(f"Sessione registrata in {audit.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
