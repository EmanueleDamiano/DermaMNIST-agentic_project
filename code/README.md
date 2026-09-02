# DermaMNIST — scheletro della pipeline agentica

Primo agente LangGraph che supervisiona classificazione e augmentation su DermaMNIST.
Copre i requisiti FR-G1 (supervisione del training), FR-G2 (pipeline end-to-end),
FR-G4 (fallback human-in-the-loop) e la traceability richiesta dal WP8.

## Struttura

```
code/
  dermaMNIST_eda.ipynb      analisi esplorativa
  eda_outputs/              contratto EDA -> training (normalizzazione, class weight, baseline)
  d4_augmentation.py        augmentation D4, con API, CLI e self-test
  training/                 layer deterministico: dataset, modello, metriche, esperimenti, inferenza
  agent/                    layer agentico: tool, modello, grafo, audit
  immagini_test/            immagini di prova, una per classe, con l'etichetta vera nel nome
  augmented/                output dell'augmentation: varianti D4 e provini d'insieme
  runs/                     un dossier per esperimento (config, metriche, log, pesi)
  agent_logs/               registro JSONL delle sessioni dell'agente
  verify_agent.py           verifica del sistema agentico, senza LLM
```

**Il principio che tiene insieme il tutto:** `training/` non importa mai `agent/`. Ogni
esperimento che l'agente lancia resta riproducibile da riga di comando senza l'agente, e ogni
fallimento e' attribuibile o al tool (riproducibile) o al modello (visibile nel log).

## Requisiti

Dipendenze gia' installate nel venv: `torch`, `torchvision`, `medmnist`, `scikit-learn`,
`langchain`, `langgraph`, `langchain-ollama`.

Serve inoltre:
- il dataset in `~/.medmnist/dermamnist.npz`;
- i pesi ImageNet in `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`;
- Ollama in esecuzione con almeno un modello che dichiari la capability `tools`.

> Nota su macOS con Python 3.14 dal framework installer: i certificati SSL non sono
> configurati, quindi i download automatici di torchvision falliscono con
> `CERTIFICATE_VERIFY_FAILED`. Il ripiego e' scaricare il checkpoint con `curl` nella cache
> di torch (entrambi i file sono gia' a posto in questo ambiente).

## Uso

### Training da riga di comando (senza agente)

```bash
python -m training.train --epochs 3 --augmentation d4 --class-weighting balanced
python -m training.train --list                # elenca i run
python -m training.train --show <run_id>       # riepilogo completo di un run
python -m training.train --epochs 1 --max-train-batches 5   # smoke test rapido
```

Un'epoca a 224 px richiede circa 50 secondi su Apple Silicon (MPS).

### Augmentation di un'immagine

```bash
python d4_augmentation.py --input immagini_test/lesione_4_vera-mel.png \
       --outdir varianti/ --contact-sheet
```

Scrive le 8 varianti D4 piu' un **provino d'insieme**: una griglia 2x4 etichettata, che e' cio'
che si apre davvero per capire a colpo d'occhio cosa e' successo. I riquadri sono ingranditi
con NEAREST, non interpolati: la D4 e' una permutazione esatta dei pixel e il provino deve
mostrarlo, non suggerire una sfocatura che non c'e'.

### Classificare un'immagine locale

Serve un run con i pesi salvati (di default `train.py` li salva, ~45 MB per run).

```bash
python -m training.predict --image immagini_test/lesione_1_vera-bcc.png
python -m training.predict --dir immagini_test --json
python -m training.predict --list-runs          # quali modelli sono utilizzabili
python -m training.predict --image foto.png --run-id 20260901-230614
```

Restituisce sempre la distribuzione completa sulle 7 classi, non solo la top-1 (FR-C2).

### Agente

```bash
python -m agent.run_cli --check                          # verifica Ollama e il modello
python -m agent.run_cli                                  # sessione interattiva
python -m agent.run_cli --task "descrivi il dataset"     # compito singolo
python -m agent.run_cli --task "classifica immagini_test/lesione_1_vera-bcc.png"
python -m agent.run_cli --task "mostrami la D4 su un campione di melanoma"
DERMA_AGENT_MODEL=qwen3.6 python -m agent.run_cli        # modello piu' capace
```

Il modello di default e' `qwen3.5:4b`, scelto per la velocita' del ciclo di sviluppo. Per la
validazione conviene un modello piu' grande via `DERMA_AGENT_MODEL`.

### Verifica

```bash
python verify_agent.py        # 44 controlli sul sistema agentico, senza LLM
python d4_augmentation.py --self-test
```

## Il modello di classificazione

ResNet-18 pre-addestrata su ImageNet, testa sostituita con un layer a 7 classi.
Le immagini 28×28 vengono portate a 224 px e normalizzate con le costanti ImageNet, cioe' il
preprocessing per cui i pesi sono stati addestrati. L'upsampling non aggiunge dettaglio: serve
a mettere l'input nella scala attesa dai pesi pre-addestrati.

Con `input_size <= 64` lo stem viene adattato automaticamente (conv 3×3 stride 1, senza
maxpool) perche' la conv 7×7 stride 2 seguita da maxpool distruggerebbe un input piccolo — al
prezzo, dichiarato in `training/model.py`, di buttare i pesi pre-addestrati del primo layer.

Riferimento: dopo **una sola epoca** a 224 px si ottengono accuracy ≈ 0,75 e AUC ≈ 0,92 sul
test, in linea con il benchmark MedMNIST v2 per ResNet-18.

### Preprocessing in inferenza

Il modello ha visto solo immagini 28×28 risalite a 224: sono intrinsecamente sfocate. Una foto
dermatoscopica nativa ad alta risoluzione, portata direttamente a 224, sarebbe molto piu' nitida
di qualsiasi immagine vista in addestramento — fuori distribuzione.

Per questo `training/predict.py` replica l'intera catena: **prima riduce a 28×28**, poi risale a
224. Butta via dettaglio di proposito. E' il prezzo di un modello addestrato a bassa risoluzione,
ed e' la ragione per cui passare alle varianti MedMNIST a 64/128/224 px e' il miglioramento con
il maggiore ritorno atteso.

## Superficie di tool dell'agente

| Tool | Cosa fa |
|---|---|
| `inspect_dataset` | numerosita', classi, sbilanciamento, class weight |
| `describe_augmentation` | cosa offre l'augmentation D4 e quando usarla |
| `augment_image` | applica la D4 a un file locale, salva varianti e provino |
| `augment_dataset_sample` | come sopra, ma su un campione del dataset scelto per classe |
| `train_model` | avvia un esperimento, restituisce un `run_id` |
| `get_run_report` | metriche di un run e confronto col baseline dell'EDA |
| `list_runs` | esperimenti gia' eseguiti |
| `list_trained_models` | quali run hanno i pesi salvati, quindi sono usabili per classificare |
| `list_images_in_directory` | elenca le immagini di una cartella locale |
| `classify_image` | classifica un'immagine, con la distribuzione su tutte e 7 le classi |
| `ask_human` | sospende e interroga l'operatore |

Due vincoli deliberati, dettati dall'uso di un modello locale: **argomenti piatti e tipizzati**
(i modelli piccoli sbagliano gli schemi annidati) e **ritorno in testo breve**, non JSON.

`train_model` passa dal gate di approvazione umana (`HumanInTheLoopMiddleware`): e' l'unico
tool che consuma minuti di calcolo.

I due tool di augmentation scrivono **solo** sotto `augmented/`, in una sottocartella derivata
dal nome dell'immagine e ridotta ai soli caratteri alfanumerici. L'agente non sceglie mai un
percorso di scrittura: un modello da pochi miliardi di parametri che allucina un path non deve
poter toccare il resto del filesystem.

## Come estenderlo

- **Nuovo tool:** una funzione in `agent/tools.py` con `@tool` sopra e `@audited` sotto, poi
  aggiungila a `TOOLS`. Se fa qualcosa di sostanziale, la logica va in `training/` e il tool
  resta un guscio.
- **Nuovo modello di classificazione:** `training/model.py`, funzione `build_model`. La
  config lo raggiunge tramite `TrainConfig`.
- **Nuova augmentation:** aggiungi il valore a `AUGMENTATIONS` in `training/config.py` e
  gestiscilo in `DermaDataset.__getitem__`. Per renderla ispezionabile dall'agente, il provino
  di `d4_augmentation.contact_sheet` funziona con qualsiasi dizionario nome -> array.
- **Verso il multi-agente:** `build_agent` restituisce un grafo LangGraph compilato, quindi
  diventa un nodo di uno `StateGraph` piu' grande senza riscritture. Conviene aprirlo quando un
  ruolo richiede un modello o un prompt sostanzialmente diverso — non prima.
- **Cambiare provider LLM:** solo `agent/llm.py`. Il grafo, i tool e l'audit non sanno quale
  modello stia girando.

## Limiti noti di questa iterazione

- Il gate di approvazione copre `train_model` e nient'altro.
- La classificazione usa un modello addestrato su 28×28: le predizioni sono indicative e non
  hanno alcun valore diagnostico.
- `InMemorySaver`: lo stato della conversazione vive quanto il processo. Per sessioni
  persistenti serve `SqliteSaver` (`langgraph-checkpoint-sqlite`).
- Nessuna misura sistematica dell'affidabilita' delle tool call: la suite di task su cui
  confrontare modelli diversi e' il passo successivo.
- Lo split ufficiale MedMNIST e' per-immagine e non per-lesione; le metriche vanno lette
  tenendo conto del possibile leakage documentato nel notebook EDA (§7).
