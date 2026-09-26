"""Registro di audit delle decisioni dell'agente (requisito di traceability, WP8).

Ogni chiamata a tool viene scritta su un file JSONL: argomenti, esito, durata, errori.
Lo scopo e' poter ricostruire una sessione senza rieseguirla.

Scelta implementativa: il logging e' esplicito, tramite il decoratore `@audited` applicato
a ogni tool, invece che agganciato ai callback del framework. Costa qualche riga in piu' ma
il registro non dipende dagli interni di LangChain, che cambiano fra versioni minori, e
resta leggibile da chiunque apra il file senza conoscere il framework.
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

AUDIT_DIR = Path(__file__).resolve().parent.parent / "agent_logs"

_current: "AuditLog | None" = None


class AuditLog:
    """Scrive eventi JSONL, uno per riga, per una singola sessione dell'agente."""

    def __init__(self, session_id: str | None = None, directory: Path | None = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.directory = directory or AUDIT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"session-{self.session_id}.jsonl"

    def event(self, kind: str, **fields: Any) -> None:
        """Registra un evento. `kind` e' il tipo: tool_call, user_message, decision, ..."""
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "kind": kind,
            **fields,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read(self) -> list[dict]:
        """Rilegge gli eventi della sessione, utile per test e per il riepilogo finale."""
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


def set_audit(log: AuditLog | None) -> None:
    """Imposta il registro attivo per la sessione corrente."""
    global _current
    _current = log


def get_audit() -> AuditLog:
    """Restituisce il registro attivo, creandone uno al volo se manca.

    Non solleva mai: un tool non deve fallire perche' il logging non e' stato inizializzato.
    """
    global _current
    if _current is None:
        _current = AuditLog()
    return _current


def audited(func: Callable) -> Callable:
    """Decoratore che registra ogni invocazione del tool: argomenti, esito, durata, errori.

    Va applicato *sotto* `@tool`, cosi' che LangChain legga firma e docstring della funzione
    originale (`functools.wraps` le preserva).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = get_audit()
        started = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            log.event(
                "tool_call",
                tool=func.__name__,
                args=kwargs or list(args),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                seconds=round(time.time() - started, 3),
            )
            raise
        log.event(
            "tool_call",
            tool=func.__name__,
            args=kwargs or list(args),
            status="ok",
            result_preview=str(result)[:500],
            seconds=round(time.time() - started, 3),
        )
        return result

    return wrapper
