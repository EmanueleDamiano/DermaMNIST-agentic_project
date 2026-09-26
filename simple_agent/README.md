# `simple_agent/` — agente LangGraph minimo per il training di FPViT

Un agente ReAct che sa lanciare un training di FPViT su DermaMNIST e leggerne
il risultato. Tre file, nessuna modifica al codice esistente: `train.py`,
`fpvit/`, `predict.py` ed `evaluate_test.py` restano intatti e vengono
chiamati dall'esterno esattamente come li chiamerebbe una persona dalla shell.

Gli agenti già presenti nella repo (`agent_tools.py`, `fpvit_runner.py`,
`langgraph_loop.py`, ora in `legacy_model_training_and_testing/`) non sono usati né importati da qui.

```
simple_agent/
  fpvit_tools.py    lancia train.py come subprocesso e comprime
                    experiment_record.json in un riassunto leggibile da un LLM.
                    Nessun import di framework agentici: cambiare framework
                    significa riscrivere agent.py, non questo file.
  agent.py          l'agente LangGraph: 3 tool, un system prompt, una CLI.
  requirements.txt  solo le dipendenze dell'agente.
```

## Setup

```bash
pip install -r simple_agent/requirements.txt
export ANTHROPIC_API_KEY=...        # oppure: ant auth login
```

## Uso

```bash
# one-shot
python -m simple_agent.agent "Allena per 2 epoche con adamw a lr 3e-4 e dimmi com'è andata"

# chat (la conversazione resta in memoria per la durata del processo)
python -m simple_agent.agent --interactive
```

I run finiscono in `runs_agent/<run_name>/`, separati da `runs/` usato a mano,
con gli stessi artefatti di sempre (`experiment_record.json`, `metrics.csv`,
`best_model.pt`, `last_model.pt`, `train.log`).

I tool si possono usare anche senza LLM, per testarli:

```python
from simple_agent.fpvit_tools import train_fpvit, list_runs, get_run_summary
print(train_fpvit(epochs=1, optimizer="adamw", lr=3e-4, run_name="probe")["summary"])
```

## Quale LLM

Il modello si sceglie con una stringa `provider:modello` (`--model`, oppure
`build_llm()` / `build_agent(model=...)`). Il provider non cambia niente
dell'agente: i tool e il system prompt sono gli stessi.

| Stringa | Cosa serve |
| --- | --- |
| `claude-opus-5` (default, prefisso opzionale) | `ANTHROPIC_API_KEY`, oppure `ant auth login` |
| `ollama:qwen3.6`, `ollama:qwen3.5:4b` | `ollama serve` attivo in locale; nient'altro, nessun costo |
| `openrouter:qwen/qwen3-235b-a22b` | `OPENROUTER_API_KEY` (`OPENROUTER_BASE_URL` se usi un gateway diverso) |
| `openai:gpt-...` | `OPENAI_API_KEY` |

```bash
python -m simple_agent.agent --model ollama:qwen3.6 "Elenca i run fatti finora"
export OPENROUTER_API_KEY=...
python -m simple_agent.agent --model openrouter:qwen/qwen3-235b-a22b "Allena 2 epoche con adamw"
```

Solo il ramo `anthropic` passa `thinking={"type": "adaptive"}`, che è un
parametro Anthropic. Il ramo `ollama` forza `num_ctx=8192`: il default di
Ollama è piccolo e **tronca in silenzio**, quindi il system prompt e il
risultato del tool cadrebbero fuori dalla finestra senza nessun errore.

### Verifica prima di fidarti: `--check`

L'unica capacità su cui poggia tutto l'agente è il **tool calling**, e il
supporto varia molto: un modello piccolo, o un gateway OpenAI-compatibile che
accetta `tools` e li ignora, risponde in prosa e non lancia mai un training —
un fallimento che sembra "l'agente ha deciso di non allenare".

```bash
python -m simple_agent.agent --model ollama:qwen3.5:4b --check
# Model: ollama:qwen3.5:4b
#   OK: the model called ['list_runs']
```

Su Ollama la capability si vede anche con `ollama show <modello>`
(sezione *Capabilities*, voce `tools`).

Cosa aspettarsi da un modello locale piccolo: chiama i tool correttamente e
riassume bene i numeri, ma sbaglia più spesso il ragionamento di dominio —
`qwen3.5:4b` ha proposto `class_weight='inverse'` **e** `balanced_sampler=True`
insieme, che il system prompt dichiara alternative e non complementari. Per
leggere le metriche va benissimo; per decidere la campagna di tuning, meno.

## I tre tool

| Tool | Cosa fa |
| --- | --- |
| `train_fpvit(...)` | valida la config, lancia `train.py`, restituisce lo stato del run: metriche all'epoca selezionata, recall per classe, classi collassate, curva di loss assottigliata, tempi |
| `list_runs()` | elenca i run già fatti, dal più recente, con il punteggio |
| `get_run_summary(run_name)` | rilegge il riassunto completo di un run precedente |

Due cose sono forzate nel tool, non lasciate al modello:

- **validazione prima del lancio**: una config non valida torna come
  `status: "invalid_config"` con i campi da correggere, senza aver avviato
  nulla (un'epoca costa ~200 s su MPS: scoprire l'errore dopo è caro);
- **budget wall-clock**: `max_seconds` di default 1800 s, tetto massimo 6 h.
  Il budget viene passato a `train.py --max-seconds`, che si ferma pulito tra
  un'epoca e l'altra e scrive comunque tutte le metriche raccolte. Se il
  processo sfora, il tool manda SIGTERM (che `train.py` gestisce salvando) e
  segnala `status: "partial"`.

Il riassunto restituito pesa circa 1 KB invece delle centinaia di KB di
`experiment_record.json` (che contiene tutta la storia per epoca e le matrici
di confusione): è quello che rende l'agente utilizzabile su run lunghi.

## Dove aggiungere harness più avanti

Tutto passa da `build_agent()` in `agent.py`, che è già parametrizzato:

```python
agent = build_agent(
    model="claude-opus-5",
    checkpointer=InMemorySaver(),   # memoria di breve termine (thread)
    store=None,                      # memoria di lungo termine tra thread
    extra_tools=[...],               # altri tool
)
```

- **memoria di breve termine** — già attiva nella CLI con `InMemorySaver`.
  Per farla sopravvivere al riavvio basta sostituirla con `SqliteSaver` o
  `PostgresSaver`: l'interfaccia è la stessa, il resto del file non cambia.
  Ogni chiamata usa `config={"configurable": {"thread_id": ...}}` (`--thread-id`).
- **memoria di lungo termine** — `store=InMemoryStore()` (poi uno store vero),
  per ricordare i risultati tra conversazioni diverse.
- **gestione del contesto** — `create_react_agent(..., pre_model_hook=...)` per
  troncare o riassumere la storia prima di ogni chiamata al modello.
- **human-in-the-loop** — `interrupt_before=["tools"]` sul grafo, per far
  approvare un training prima che parta.
- **altri tool** — `extra_tools=[...]`: la valutazione sul test set
  (`evaluate_test.py`) e l'inferenza (`predict.py`) si avvolgono allo stesso
  modo di `train_fpvit`, aggiungendo una funzione in `fpvit_tools.py` e un
  wrapper di dieci righe in `agent.py`.

## Cosa sa l'agente (system prompt)

Le informazioni di dominio che non sono deducibili dalla firma dei tool stanno
nel system prompt: lo sbilanciamento del train split
(`[228, 359, 769, 80, 779, 4693, 99]`, la classe 5 è il 67 % — quindi mai
giudicare un run dall'accuracy), il costo di un'epoca, la differenza tra
`sgd` (ricetta del paper) e `adamw`, il fatto che `class_weight` e
`balanced_sampler` sono alternative e non complementari, e che il test split
non si tocca.
