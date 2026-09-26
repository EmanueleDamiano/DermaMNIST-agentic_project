# `webapp/`: la piattaforma DermaAgent

Punto di accesso all'orchestratore (testing agent → reviewer → umano), con la
vista live di quale agente sta lavorando e su cosa. Lo stile riprende quello di
`platform/app.py` del progetto AgenticDerma: chat al centro, drawer
**Processo** con il grafo degli agenti, la traccia della comunicazione e lo
stato della memoria.

```bash
python -m webapp               # http://127.0.0.1:8000
python -m webapp --port 8080
```

La cartella si chiama `webapp/` e non `platform/` perché un pacchetto
`platform` nella root del progetto coprirebbe il modulo `platform` della
libreria standard, che torch importa all'avvio.

## Cosa si vede

- **Chat**: si allegano una o più immagini (fino a 16, massimo 10 MB l'una)
  oppure si sceglie un campione di `test_samples/`. Il testo è un contesto
  facoltativo per gli agenti. Per ogni immagine compare una scheda con:
  - classe finale, confidenza e verdetto del reviewer (✓ coerente / ⚠ criticità);
  - le barre del voto pesato dell'ensemble;
  - la sintesi del reviewer e il modello decisivo;
  - le criticità con la loro gravità, la memoria e le fonti.
  
  Con "Racconta il processo" attivo, la risposta si apre con **Cosa è
  successo**: il racconto del reviewer, passo per passo.
- **Processo** (grafo): Orchestrator in alto; Testing agent, Reviewer agent e
  Human sotto. Ogni agente mostra i propri nodi LangGraph, che si accendono
  mentre girano:
  - Testing agent: `load_inputs → … → reason → write_log`;
  - Reviewer agent: `gather → review → render`.
  
  Gli archi si animano quando il lavoro passa da un agente all'altro. Human
  risulta "non richiesto" se la conferma umana è spenta, e "attende te" se il
  grafo è fermo sull'`interrupt()`.
- **Comunicazione**: un evento per ogni nodo completato, con un riassunto
  leggibile. I "dettagli" contengono la traccia completa del nodo, cioè le
  stesse righe che `-v` stampa da terminale.
- **Memoria e modelli**: righe nei due log, KB, ensemble con la skill di ogni
  modello.
- **Agenti** (in alto): LLM del tester e del reviewer (Ollama rilevato in
  automatico, Claude se c'è `ANTHROPIC_API_KEY`, oppure nessuno), "Racconta il
  processo", "Chiedimi conferma sulle criticità", soglia per escludere i
  modelli deboli. Le scelte restano nel browser (localStorage).
- **Decisione umana**: se il reviewer segnala criticità e la conferma è
  attiva, si apre una finestra con le criticità e le scelte *Rifiuta /
  Accetta* (con nota facoltativa). "Decido dopo" la chiude: il grafo resta in
  pausa e la finestra si riapre da **Processo**.

## Explainability: le due finestre

Si aprono in una scheda separata (`window.open`), così la pagina principale
resta leggera e i dati si caricano solo quando servono.

**Contesto del ragionamento** (`/context?job=…&step=tester.reason|reviewer.review`).
Si apre cliccando **Ragionamento** (Testing agent) o **Verifica di coerenza**
(Reviewer) nel grafo, dal link "Apri il contesto completo ↗" nella traccia,
oppure da "Contesto del ragionamento ↗" sotto la risposta. Mostra **tutto
quello che l'LLM riceve**, anche mentre sta ancora ragionando (la pagina si
aggiorna ogni 1.5 s finché il passo è in corso):

- **Vista leggibile**:
  - per il tester: ensemble con skill e classi mai riconosciute; per ogni
    immagine soft vote, hard vote, opinione di ogni modello, candidati per
    l'override, memoria (stessa immagine) e i passaggi della KB, evidenziati
    in verde quelli poi citati nella risposta;
  - per il reviewer: la motivazione da verificare, il voto, la quota di ogni
    modello sulla classe finale, i controlli automatici, i passaggi citati dal
    tester, quelli recuperati in modo indipendente e la traccia completa.
- **Prompt di sistema** e **Payload JSON**: i testi esatti inviati, copiabili.
  In testa: numero di caratteri e token stimati.
- **Risposta**: la decisione validata e la risposta grezza dell'LLM prima della
  validazione in codice (incluso un eventuale override rifiutato).

Senza LLM la finestra mostra gli stessi dati, cioè quelli su cui ha lavorato
la regola deterministica.

Come è possibile vederlo *prima* che il nodo finisca: il payload è costruito
da due funzioni a livello di modulo, `predict_agent.graph.build_reason_payload`
e `review_agent.graph.build_review_payload`. Le usano sia gli agenti sia la
piattaforma, che le applica all'input del nodo trasmesso dallo stream
LangGraph all'avvio del nodo. È verificato che il payload ricostruito sia
identico a quello inviato all'LLM.

**Knowledge base** (`/kb`). Si apre cliccando **Memoria da retrieval** nel
pannello Processo. Mostra:
- statistiche del corpus e metodo di retrieval;
- un pulsante per ognuna delle 7 classi, che esegue *la stessa query espansa*
  usata dal Testing agent e mostra i passaggi in ordine di punteggio, con i
  termini della query evidenziati;
- una ricerca libera con la stessa funzione `KnowledgeBase.search`;
- l'elenco delle fonti, con tutti i passaggi (caricati solo quando si apre una
  fonte).

## Training dalla piattaforma

- **Pulsante "Training"**: obiettivo in parole, architettura ("decide l'agente",
  cioè FPViT salvo proposta diversa, oppure una delle quattro), autonomia, run
  massimi, minuti, epoche per run, trigger di plateau, metrica, obiettivo, LLM
  del training agent.
- **Richiesta scritta senza immagini** (per esempio "vorrei migliorare il
  riconoscimento del dermatofibroma"): la instrada l'orchestratore, con l'LLM
  del training agent. Se non è chiara, si apre la finestra "Cosa vuoi fare?".
- **Grafo**: in una campagna al posto di tester e reviewer compare la scheda
  del Training agent con i suoi 13 passi, che si ripetono a ogni run.
- **"Training live"**: una scheda per run con la curva della balanced accuracy
  di validazione, epoca per epoca (dallo stream `custom` del nodo `train`), e
  la loss di training tratteggiata. Sotto, la tabella dei run: cosa ha fatto e
  perché, configurazione, motivazione dell'augmentation, risultato, flag della
  diagnosi e verdetto.
- **Una finestra umana unica, che cambia secondo l'interruzione**:
  - criticità del reviewer;
  - chiarimento dell'intento;
  - piano, con i campi modificabili;
  - proposta di run: iperparametri, augmentation effettiva e motivazione, cosa
    cambia, perché serve l'umano, rilievi del training reviewer; si può
    approvare, chiedere un'altra proposta con feedback o fermare;
  - promozione: ensemble con e senza il candidato e recall per classe.
- **Un blocco separato per il training**: una campagna di ore non ferma le
  previsioni.

## Come funziona

Né gli agenti né l'orchestratore sono stati scritti per questa UI.
`webapp/app.py` esegue l'orchestratore con

```python
app.stream(inputs, config, stream_mode=["tasks"], subgraphs=True)
```

LangGraph emette l'inizio e il risultato di **ogni nodo**: quelli
dell'orchestratore e quelli dei grafi di tester e reviewer che girano al suo
interno, ognuno con il proprio namespace (`tester:…`, `reviewer:…`). Il server
li traduce nello stato del grafo e nella traccia; la pagina interroga
`/api/status` ogni 0.6 s.

La pausa umana è l'`interrupt()` dell'orchestratore. Il server la rileva con
`get_state()`, la pagina risponde su `/api/resume` e il job riparte con
`Command(resume={...})` sullo stesso `thread_id`.

| Endpoint | |
| --- | --- |
| `GET /` | la pagina (`page.html`) |
| `GET /api/health` | modelli, KB, LLM disponibili, righe nei log, campioni (cache 30 s) |
| `POST /api/run` | `{samples, uploads, request, settings}` → `{job}` |
| `GET /api/status?id=` | stato del grafo, nodi, traccia, risultato |
| `POST /api/resume` | `{job, decision, note}`: risponde all'interrupt |
| `GET /api/image?job=&i=` / `GET /api/sample?name=` | le immagini analizzate / i campioni |
| `GET /context`, `GET /api/context?job=&step=` | finestra e dati del contesto di ragionamento |
| `GET /kb`, `GET /api/kb`, `GET /api/kb/search?q=&k=` | finestra, corpus completo, ricerca |
| `POST /api/run` con `mode: "train"` o solo `request` | campagna di training / richiesta da instradare (`train`: vincoli della campagna) |
| `POST /api/resume` | `{job, decision, note, edits}`: `decision` tra le `options` dell'interruzione, `edits` solo per il piano |

**Scelte:**
- **Un processo alla volta** (`RUN_LOCK`): modelli e LLM sono condivisi. Un
  secondo job resta in coda e lo dice nella traccia.
- **Un orchestratore compilato per ogni combinazione di impostazioni**,
  riusato tra i job, con un `InMemorySaver` (serve per `get_state` e per
  l'interrupt).
- **Solo immagini caricate o campioni**: nessun path arriva dal browser al
  filesystem. Gli upload finiscono in `runs_predict/uploads/<job>/`.
- **Server solo su `127.0.0.1`.** Nessuna autenticazione, quindi non va esposto
  in rete così com'è.

## Verificato (2026-09-23, Chrome integrato, macOS)

- Senza LLM, tutte le 7 immagini con conferma umana: il grafo avanza fino a
  "attende te", la finestra mostra le 4 immagini segnalate; con *Accetta* e una
  nota il grafo riparte e completa. Le 7 schede e la decisione sono salvate in
  `reviews_log.jsonl`.
- Con `ollama:qwen3.6` su un'immagine: il grafo mostra dal vivo il Testing
  agent fermo su "Ragionamento" (circa 2 minuti) e poi il Reviewer su "Verifica
  di coerenza" (circa 2 minuti). La traccia ha tutti i 15 eventi e la risposta
  si apre con il racconto del processo.

## Verificato (2026-09-25, training)

- Campagna dal pulsante Training (ResNet-18, 2 run, politica deterministica):
  - piano mostrato e modificato (budget 6 → 7 minuti, applicato);
  - primo run approvato dalla finestra;
  - curve live a 37 s/epoca;
  - secondo run (`class_weight` inverse, per il dermatofibroma a recall 0) chiesto all'umano da `guarded`;
  - promozione +0.0109 mostrata e rifiutata; scheda finale "non promosso".
- Previsione di un campione dalla UI: instradata come "previsione (ci sono
  immagini)", stesso risultato di prima.
- Testo senza immagini e senza LLM: finestra "Cosa vuoi fare?" → Annulla →
  "Richiesta annullata.".
- Le cartelle e le righe di log create dai test sono state rimosse.
