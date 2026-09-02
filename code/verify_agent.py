"""Verifica del sistema agentico, eseguibile senza LLM.

    python verify_agent.py

Copre tre livelli, dal basso verso l'alto:

1. **Tool da soli** — invocati direttamente, senza grafo e senza modello.
2. **Grafo con modello finto** — `ScriptedChatModel` sostituisce l'LLM, quindi un fallimento
   qui e' necessariamente del grafo o dei tool, mai del modello.
3. **Sospensioni** — approvazione umana prima di `train_model` e domanda posta con `ask_human`.
4. **Augmentation** — esecuzione della D4 su file e su campioni del dataset.
5. **Classificazione** — inferenza su immagini locali, se esiste un modello con pesi salvati.

Non esegue training reali: il percorso di approvazione viene verificato con un rifiuto, che
attraversa lo stesso codice senza consumare minuti di calcolo.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.audit import AuditLog, set_audit
from agent.fake_model import ScriptedChatModel, final_message, tool_call_message
from agent.graph import build_agent
from agent.tools import TOOLS

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'OK  ' if condition else 'FALLITO'} {name}" + (f" — {detail}" if detail else ""))


def tool_by_name(name: str):
    return next(t for t in TOOLS if t.name == name)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------------------
def test_tools_standalone() -> None:
    section("1. Tool invocati direttamente, senza grafo e senza modello")

    out = tool_by_name("inspect_dataset").invoke({})
    check("inspect_dataset riporta i conteggi noti dall'EDA", "7007" in out and "58" in out)

    out = tool_by_name("describe_augmentation").invoke({})
    check("describe_augmentation elenca le 8 varianti D4", "r90" in out and "m_r270" in out)

    out = tool_by_name("list_runs").invoke({})
    check("list_runs risponde senza sollevare eccezioni", isinstance(out, str) and len(out) > 0)

    # Percorsi di errore: devono restituire un messaggio, non far cadere l'agente.
    out = tool_by_name("train_model").invoke({"epochs": 99})
    check("train_model rifiuta epochs fuori intervallo", "Errore" in out and "1-10" in out)

    out = tool_by_name("train_model").invoke({"epochs": 1, "augmentation": "inesistente"})
    check("train_model rifiuta un'augmentation sconosciuta", "Errore" in out)

    out = tool_by_name("get_run_report").invoke({"run_id": "run_che_non_esiste"})
    check("get_run_report segnala un run_id inesistente", "Errore" in out)


# --------------------------------------------------------------------------------------
def test_graph_with_fake_model() -> None:
    section("2. Grafo end-to-end con modello finto")

    model = ScriptedChatModel(
        responses=[
            tool_call_message("inspect_dataset", {}),
            final_message("Il dataset ha 7007 immagini di training su 7 classi."),
        ]
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver(), require_approval=False)
    config = {"configurable": {"thread_id": "test-1"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "descrivi il dataset"}]}, config=config
    )

    kinds = [type(m).__name__ for m in result["messages"]]
    check("il grafo esegue la tool call e produce un ToolMessage", "ToolMessage" in kinds, str(kinds))
    tool_output = next(
        (m.content for m in result["messages"] if type(m).__name__ == "ToolMessage"), ""
    )
    check("il ToolMessage contiene l'output reale del tool", "7007" in tool_output)
    check("il modello e' stato interrogato piu' di una volta", len(model.calls) >= 2,
          f"{len(model.calls)} chiamate")


# --------------------------------------------------------------------------------------
def test_approval_gate() -> None:
    section("3a. Gate di approvazione umana prima di train_model")

    model = ScriptedChatModel(
        responses=[
            tool_call_message("train_model", {"epochs": 1, "augmentation": "d4"}),
            final_message("Il training e' stato rifiutato dall'operatore."),
        ]
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver(), require_approval=True)
    config = {"configurable": {"thread_id": "test-2"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "addestra il modello"}]}, config=config
    )
    check("il grafo si sospende invece di addestrare", "__interrupt__" in result)

    payload = result["__interrupt__"][0].value if "__interrupt__" in result else {}
    check("la sospensione descrive l'azione da approvare", "action_requests" in payload, str(payload)[:120])
    if "action_requests" in payload:
        action = payload["action_requests"][0]
        check("l'azione e' proprio train_model con i suoi argomenti",
              action.get("name") == "train_model" and action.get("args", {}).get("epochs") == 1,
              str(action))

    # Rifiuto: attraversa lo stesso percorso dell'approvazione senza spendere minuti di calcolo.
    resumed = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "non ora"}]}), config=config
    )
    tool_messages = [m for m in resumed["messages"] if type(m).__name__ == "ToolMessage"]
    check("il rifiuto impedisce l'esecuzione del training",
          any("rejected" in str(m.content).lower() for m in tool_messages),
          str([str(m.content)[:60] for m in tool_messages]))
    check("nessun run e' stato creato dal rifiuto",
          not any("run_id=" in str(m.content) for m in tool_messages))


# --------------------------------------------------------------------------------------
def test_ask_human() -> None:
    section("3b. Domanda all'operatore con ask_human")

    model = ScriptedChatModel(
        responses=[
            tool_call_message("ask_human", {"question": "Quante epoche vuoi?"}),
            final_message("Procedo con 3 epoche."),
        ]
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver(), require_approval=False)
    config = {"configurable": {"thread_id": "test-3"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "addestra come preferisci"}]}, config=config
    )
    check("ask_human sospende il grafo", "__interrupt__" in result)
    payload = result["__interrupt__"][0].value if "__interrupt__" in result else {}
    check("la sospensione trasporta la domanda dell'agente",
          payload.get("type") == "question" and "epoche" in payload.get("question", ""),
          str(payload)[:120])

    resumed = agent.invoke(Command(resume="tre epoche"), config=config)
    tool_output = next(
        (m.content for m in resumed["messages"] if type(m).__name__ == "ToolMessage"), ""
    )
    check("la risposta umana torna all'agente", "tre epoche" in str(tool_output), str(tool_output))


# --------------------------------------------------------------------------------------
def test_augmentation_tools() -> None:
    section("4. Esecuzione dell'augmentation")

    import numpy as np
    from d4_augmentation import contact_sheet, expand_d4
    from agent.tools import AUGMENTED_DIR

    # --- la funzione pura ---
    img = np.random.default_rng(0).integers(0, 256, size=(28, 28, 3), dtype=np.uint8)
    sheet = contact_sheet(expand_d4(img), title="test")
    check("contact_sheet produce un'immagine di dimensione plausibile",
          sheet.width > 400 and sheet.height > 250, f"{sheet.width}x{sheet.height}")

    # I riquadri devono essere ingranditi SENZA interpolazione: con un fattore 4 ogni pixel
    # dell'originale diventa un blocco 4x4 uniforme. Se qualcuno passasse a bilineare, questo
    # controllo fallirebbe.
    # Geometria del provino: banda del titolo 30 px, padding 10 px, fattore di ingrandimento 4
    # (da 28 a 112). Il pixel sorgente (2,2) occupa quindi il blocco 4x4 che inizia a
    # (y=30+10+8, x=10+8). Campionarlo altrove finirebbe nel titolo o nel padding.
    pixels = np.asarray(sheet)
    y0, x0 = 30 + 10 + 8, 10 + 8
    block = pixels[y0 : y0 + 4, x0 : x0 + 4]
    check("l'ingrandimento e' a blocchi, non interpolato",
          bool((block == block[0, 0]).all()), f"blocco unico: {block[0,0].tolist()}")

    # --- i tool ---
    out = tool_by_name("augment_image").invoke(
        {"image_path": "/percorso/inesistente.png"})
    check("augment_image segnala un file inesistente", "Errore" in out)

    out = tool_by_name("augment_image").invoke(
        {"image_path": "immagini_test/lesione_4_vera-mel.png", "variant": "r45"})
    check("augment_image rifiuta una variante inventata", "Errore" in out and "r90" in out)

    out = tool_by_name("augment_dataset_sample").invoke({"class_name": "classe_inesistente"})
    check("augment_dataset_sample elenca le classi ammesse", "Errore" in out and "akiec" in out)

    out = tool_by_name("augment_dataset_sample").invoke({"class_name": "df", "index": 9999})
    check("augment_dataset_sample segnala un indice fuori intervallo", "Errore" in out)

    sample = Path("immagini_test/lesione_4_vera-mel.png")
    if not sample.exists():
        check("augmentation reale (saltata: manca immagini_test/)", True)
        return

    out = tool_by_name("augment_image").invoke({"image_path": str(sample)})
    written = [line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("  ")]
    check("augment_image genera le 8 varianti", len(written) == 8, f"{len(written)} file")
    check("i file dichiarati esistono davvero", all(Path(p).exists() for p in written))
    check("augment_image produce il provino d'insieme", "Provino" in out)

    out = tool_by_name("augment_dataset_sample").invoke({"class_name": "mel", "index": 0})
    check("augment_dataset_sample funziona su una classe del dataset",
          "Varianti generate: 8" in out and "Errore" not in out)

    # Una singola variante non deve generare un provino: una griglia di una cella e' inutile.
    out = tool_by_name("augment_image").invoke({"image_path": str(sample), "variant": "r180"})
    check("una variante singola non genera provino",
          "Varianti generate: 1" in out and "Provino" not in out)

    # Contenimento: nessuna scrittura deve uscire da augmented/, nemmeno con un nome ostile.
    all_written = list(AUGMENTED_DIR.rglob("*"))
    check("ogni file scritto resta dentro augmented/",
          all(p.resolve().is_relative_to(AUGMENTED_DIR.resolve()) for p in all_written),
          f"{len(all_written)} percorsi controllati")

    out = tool_by_name("augment_image").invoke(
        {"image_path": "../../../immagini_test/lesione_4_vera-mel.png"})
    check("un percorso con '..' non fa scrivere fuori", "Errore" in out or "augmented" in out)


# --------------------------------------------------------------------------------------
def test_classification_tools() -> None:
    section("5. Classificazione di immagini locali")

    from training.predict import find_runs_with_weights

    out = tool_by_name("list_trained_models").invoke({})
    has_model = bool(find_runs_with_weights())
    check("list_trained_models riflette la presenza di pesi salvati",
          ("Modelli disponibili" in out) == has_model, out[:70])

    # Percorsi di errore: devono restituire un messaggio, non sollevare eccezioni.
    out = tool_by_name("classify_image").invoke({"image_path": "/percorso/inesistente.png"})
    check("classify_image segnala un file inesistente", "Errore" in out)

    out = tool_by_name("classify_image").invoke({"image_path": "."})
    check("classify_image segnala che il percorso e' una directory",
          "Errore" in out and "directory" in out)

    out = tool_by_name("list_images_in_directory").invoke({"directory": "/non/esiste"})
    check("list_images_in_directory segnala una directory inesistente", "Errore" in out)

    if not has_model:
        check("classificazione reale (saltata: nessun modello addestrato)", True,
              "addestra un modello per attivare questo controllo")
        return

    sample_dir = Path("immagini_test")
    if not sample_dir.is_dir():
        check("classificazione reale (saltata: manca immagini_test/)", True)
        return

    out = tool_by_name("list_images_in_directory").invoke({"directory": str(sample_dir)})
    check("list_images_in_directory elenca le immagini", "Trovate" in out and ".png" in out)

    sample = sorted(sample_dir.glob("*.png"))[0]
    out = tool_by_name("classify_image").invoke({"image_path": str(sample)})
    check("classify_image restituisce una classe con confidenza",
          "Classe predetta" in out and "confidenza" in out, out.split(chr(10))[1][:70])
    check("classify_image dichiara il limite del modello", "non ha valore diagnostico" in out)

    # Le probabilita' devono essere una distribuzione: FR-C2 chiede la distribuzione completa.
    from training.predict import classify_file
    result = classify_file(sample)
    total = sum(result["probabilities"].values())
    check("le probabilita' coprono tutte e 7 le classi", len(result["probabilities"]) == 7)
    check("le probabilita' sommano a 1", abs(total - 1.0) < 1e-4, f"somma={total:.6f}")

    # Un'immagine ad alta risoluzione deve passare per la stessa catena, senza errori.
    big = sample_dir / "lesione_alta_risoluzione_vera-mel.png"
    if big.exists():
        out = tool_by_name("classify_image").invoke({"image_path": str(big)})
        check("classify_image gestisce immagini ad alta risoluzione", "Classe predetta" in out)


# --------------------------------------------------------------------------------------
def test_audit_log() -> None:
    section("6. Registro di audit")

    with tempfile.TemporaryDirectory() as tmp:
        audit = AuditLog(session_id="test", directory=Path(tmp))
        set_audit(audit)

        tool_by_name("inspect_dataset").invoke({})
        tool_by_name("get_run_report").invoke({"run_id": "inesistente"})

        events = audit.read()
        check("ogni tool call e' registrata", len(events) == 2, f"{len(events)} eventi")
        check("il registro riporta il nome del tool",
              {e["tool"] for e in events} == {"inspect_dataset", "get_run_report"})
        check("il registro riporta argomenti, esito e durata",
              all({"args", "status", "seconds"} <= set(e) for e in events))
        check("il registro e' JSONL rileggibile riga per riga",
              all(e["kind"] == "tool_call" for e in events))

    set_audit(None)


# --------------------------------------------------------------------------------------
def main() -> int:
    print("Verifica del sistema agentico DermaMNIST (senza LLM)")
    test_tools_standalone()
    test_graph_with_fake_model()
    test_approval_gate()
    test_ask_human()
    test_augmentation_tools()
    test_classification_tools()
    test_audit_log()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\nEsito: {passed}/{total} controlli superati")
    if passed < total:
        print("\nFalliti:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name} {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
