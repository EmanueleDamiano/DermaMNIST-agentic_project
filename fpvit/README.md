# FPViT per DermaMNIST — baseline di classificazione

Implementazione da zero del **Feature Pyramid Vision Transformer (FPViT)**
(Liu, Li, Cao, Liu, Cao — IJCNN 2022) applicato a **DermaMNIST**, pensata
come primo classificatore candidato per il progetto **AgenticDerma**
(WP3 — Training, WP4 — Evaluation).

Non è stato trovato un repository ufficiale pubblico per questo paper: il
codice qui presente è una reimplementazione originale, scritta a partire
dall'architettura e dalle equazioni descritte nel paper (Sezione III),
testata con forward/backward pass su dati sintetici.

## Architettura

```
input 3x28x28
   │
 stem conv 3x3 (no downsampling, adattato per immagini 28x28)
   │
 layer1 (2 BasicBlock, stride 1)  -> B1: 64 x28x28  -> ViT head 1 -> a1 (dim=D)
   │
 layer2 (2 BasicBlock, stride 2)  -> B2: 128x14x14   -> ViT head 2 -> a2 (dim=D)
   │
 layer3 (2 BasicBlock, stride 2)  -> B3: 256x 7x 7   -> ViT head 3 -> a3 (dim=D)
   │
 layer4 (2 BasicBlock, stride 2)  -> B4: 512x 4x 4   -> pool+fc    -> a4 (dim=D)

 A = concat(a1, a2, a3, a4)  ->  Linear(4D -> num_classes)  ->  logits
```

- **Estrattore multi-scala**: ResNet-18 con lo stem "small-image" (conv 3x3
  stride 1, senza il max-pool iniziale usato per ImageNet), coerente con la
  Table I del paper e con le dimensioni 28→28→14→7→4 riportate.
- **Teste ViT poco profonde** (una per B1, B2, B3): ogni posizione spaziale
  della feature map è trattata come una "patch" 1×1, proiettata linearmente
  a dimensione `embed_dim`, con class token e position embedding appresi,
  seguita da un encoder Transformer pre-norm (default: 4 layer, come nel
  paper) — corrisponde alle Eq. 2–3 del paper.
- **Testa ResNet** su B4: global average pooling + layer lineare (l'ablation
  del paper mostra che tenere questa testa, "4 heads", batte la variante
  "3 heads" che la scarta — disponibile con `--no-resnet-head`).
- **Fusione finale**: concatenazione dei 4 vettori di attivazione + layer
  fully-connected (Eq. 5–6 del paper). Per DermaMNIST (multi-classe,
  singola etichetta, 7 classi) si usa `CrossEntropyLoss`, come indicato nel
  paper per il task binary/multi-class.

File principali:

| File | Contenuto |
| --- | --- |
| `fpvit/model.py` | `BasicBlock`, `ResNet18Extractor`, `ShallowViTHead`, `ResNetHead`, `FPViT`, `build_fpvit()` |
| `fpvit/dataset.py` | Caricamento DermaMNIST (pacchetto `medmnist`), applica l'augmentation al solo train split |
| `fpvit/augment.py` | `AugmentationConfig`, `PRESETS`, `AUG_SEARCH_SPACE`, `resolve_augmentation()`, `build_train_transform()` |
| `fpvit/engine.py` | Loop di training/valutazione, metriche (ACC, AUC macro, F1 macro, balanced accuracy, matrice di confusione) |
| `train.py` | Script CLI di training, produce checkpoint + `experiment_record.json` |
| `evaluate_test.py` | Valutazione isolata sul test set a partire da un checkpoint congelato |
| `predict.py` | Inferenza su una singola immagine (o cartella), con scoring opzionale su `labels.csv` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verificato con Python 3.14 / torch 2.14 / medmnist 3.0.2 su macOS arm64.
`--device auto` (default) sceglie cuda, poi mps (Apple Silicon), poi cpu.

## Verifica effettuata

Eseguita end-to-end il 2026-09-16 su macOS arm64 (M-series, MPS) con il
vero DermaMNIST scaricato da `medmnist`:

- **Dati**: split ufficiale confermato — train 7007 / val 1003 / test 2005,
  7 classi, 3 canali. I valori `mean`/`std` in `dataset.py` sono stati
  ricalcolati sul train split reale e risultano corretti entro 0.005
  (attuali: mean `[0.7636, 0.5372, 0.5614]`, std `[0.1362, 0.1540, 0.1687]`).
- **`test_samples/`**: le 7 PNG sono state confrontate pixel-per-pixel con
  il test split di `dermamnist.npz` agli indici indicati in `labels.csv` —
  corrispondenza esatta, etichette corrette.
- **Modello**: forward pass OK, output `(batch, 7)`, 16 897 543 parametri.
- **Training reale**: 3 epoche (`--optimizer adamw --lr 3e-4`) →
  train_loss 0.963 → 0.846 → 0.781, val macro-AUC 0.829 → 0.889 → 0.894.
  Il modello impara; 3 epoche non sono un risultato, solo una prova di
  funzionamento.
- **`evaluate_test.py`**: eseguito una volta sul checkpoint usa-e-getta a 3
  epoche per verificarne la meccanica (report scritto fuori da `runs/`, non
  è un risultato riportabile).
- **`predict.py`**: 3/7 corrette sulle immagini di `test_samples/` con il
  checkpoint a 3 epoche, coerente con la balanced accuracy di 0.32.

### Costo computazionale misurato

Su MPS (batch 128): **~3.1 s/batch → ~200 s per epoca** (train + val), cioè
**~5.5 h per le 100 epoche di default**. Il collo di bottiglia è la testa
ViT su B1: a 28×28 senza downsampling sono 785 token per 4 layer
Transformer, più di tutte le altre teste sommate.

```bash
pip install -r requirements.txt
python train.py --epochs 100 --batch-size 128 --optimizer sgd --lr 1e-3 --out runs/fpvit_run1
python evaluate_test.py --checkpoint runs/fpvit_run1/best_model.pt
python predict.py --image test_samples/05_melanoma.png --checkpoint runs/fpvit_run1/best_model.pt
```

Il primo avvio scarica automaticamente `dermamnist.npz` in `~/.medmnist`
tramite il pacchetto `medmnist`. Se il tuo ambiente non ha accesso diretto a
Zenodo, scarica il file manualmente e passa `--data-root <cartella>`.

## Augmentation

Applicata **solo al train split**; val e test ricevono sempre solo
`ToTensor` + `Normalize`. Tutto è descritto da un `AugmentationConfig`
piatto e serializzabile in JSON (`fpvit/augment.py`).

| Campo | Default | Note |
| --- | --- | --- |
| `horizontal_flip` | 0.5 | probabilità |
| `vertical_flip` | 0.5 | probabilità |
| `rotate_90` | `True` | rotazione di un multiplo di 90°, **esatta**: nessuna interpolazione né fill |
| `rotation_degrees` | 0.0 | rotazione ad angolo arbitrario; 0 la disattiva |
| `random_resized_crop` | `True` | traslazione + scaling senza alcun bordo artificiale |
| `rrc_scale_min` / `rrc_scale_max` | 0.8 / 1.0 | frazione di area conservata |
| `rrc_ratio_min` / `rrc_ratio_max` | 0.9 / 1.111 | jitter di aspect ratio |
| `color_jitter` | `True` | attiva i quattro campi `cj_*` |
| `cj_brightness` / `cj_contrast` / `cj_saturation` | 0.1 | leggeri |
| `cj_hue` | 0.02 | **volutamente minuscolo**: la tinta è segnale diagnostico |
| `cutout` | `False` | dalla ricetta del paper, ma è un regolarizzatore |
| `cutout_size` / `cutout_p` | 8 / 0.5 | |

Flip orizzontale + verticale + `rotate_90` generano il **gruppo diedrale
completo (8 orientamenti)**. Sono esattamente label-preserving: le immagini
dermatoscopiche non hanno orientamento canonico. È la diversità più
economica disponibile, e conta soprattutto sul ramo lungo — la classe 3 ha
80 immagini di training.

### Preset

| Preset | Contenuto |
| --- | --- |
| `none` | niente — per diagnosticare underfitting o misurare quanto costa l'augmentation |
| `dihedral` | solo le simmetrie esatte: nessuna interpolazione, nessun fill, nessun cambio di colore |
| `default` | `dihedral` + `RandomResizedCrop(0.8–1.0)` + color jitter leggero |
| `strong` | range più ampi, rotazione ±15°, + Cutout — per run lunghi dove l'overfitting è comparso |
| `paper` | lettura coerente della ricetta del paper (flip orizzontale + Cutout), **senza** il crop con fill nero |

### Tre modi di specificarla, in ordine di precedenza

```bash
python train.py --aug-preset dihedral
python train.py --aug-config '{"horizontal_flip":0.5,"cj_hue":0.01,...}'
python train.py --aug-config runs/fpvit_run1/experiment_record.json   # ripete quel run
python train.py --aug-preset default --aug-set cj_hue=0.01 --aug-set cutout=true
```

`--aug-config` accetta JSON inline, un path a un file JSON, o direttamente un
`experiment_record.json` precedente (ne legge la chiave `augmentation`).
`--aug-set` è ripetibile e ha la precedenza su tutto.

La config **risolta** viene salvata campo per campo sotto `augmentation` in
`experiment_record.json` e in entrambi i checkpoint — non il solo nome del
preset, perché il contenuto di un preset può cambiare tra versioni.

### Uso programmatico / da agente

```python
from fpvit.augment import (AUG_SEARCH_SPACE, AugmentationConfig,
                            preset, resolve_augmentation)

cfg = preset("dihedral")
cfg = cfg.replace(cj_hue=0.01, cutout=True)     # ritorna una config validata
cfg = AugmentationConfig.from_dict({...})        # rifiuta le chiavi ignote
cfg = AugmentationConfig.sample()                # random search
print(cfg.describe())                            # ops effettivamente applicate
```

`AUG_SEARCH_SPACE` è il contratto leggibile da macchina: per ogni campo
dichiara `type`, `low`/`high` (o `choices`), `default` e un campo `note` con
la **ragione di dominio** del bound — così un agente che propone una config
ha il ragionamento a disposizione invece di doverlo riscoprire. Passalo a
Optuna per una ricerca sample-efficient, o usa `AugmentationConfig.sample()`
per una random search.

`from_dict` **rifiuta le chiavi ignote** invece di ignorarle, e `validate()`
solleva `ValueError` nominando il campo fuori range. Per una ricerca
automatica è essenziale: una chiave scritta male che viene scartata in
silenzio produce un run che sembra configurato e non lo è.

### Cosa è stato rimosso, e perché

`RandomCrop(28, padding=4)`. I default di torchvision sono
`fill=0, padding_mode='constant'`: su un dataset la cui media è
`(195, 137, 143)` iniettava un bordo **nero puro** in un'immagine di pelle
chiara. Peggio, imitava un artefatto **reale**: parte di HAM10000 ha
vignettatura dermatoscopica (angoli scuri), quindi l'augmentation insegnava
al modello che i bordi scuri sono rumore ignorabile proprio dove sono un
confondente correlato allo strumento.

`RandomResizedCrop` lo sostituisce: campiona una regione *interna*
all'immagine, quindi traslazione e scaling non richiedono alcun fill. L'unica
operazione che non può evitare un bordo (rotazione ad angolo arbitrario)
riempie col colore medio del dataset, non col nero.

Il Cutout è stato **conservato** — è implementato correttamente (riempie con
`0.0` *dopo* la normalizzazione, cioè col colore medio, come in DeVries &
Taylor) e fa parte della ricetta del paper — ma è ora disattivato per
default: è un regolarizzatore, e nel run a 10 epoche non c'era alcun
overfitting da regolarizzare.

## Monitoraggio del training

`train.py` riscrive **a ogni epoca** tutti gli artefatti sotto `<out>/`, quindi
un run interrotto conserva tutto il log fino a quel punto (le scritture JSON/CSV
sono atomiche, tmp + rename):

| File | Contenuto |
| --- | --- |
| `experiment_record.json` | config, seed, num_params, device, `best_epoch`, `best_score`, `selection_metric`, `epochs_completed`, `stop_reason`, pesi/conteggi per classe e la `history` per epoca — inclusi precision/recall/F1 **per classe** e la matrice di confusione di validazione |
| `metrics.csv` | una riga piatta per epoca (`epoch, lr, train_loss, train_acc, val_loss, val_acc, val_macro_auc, val_macro_f1, val_balanced_acc, epoch_seconds`), pronta da plottare o da seguire con `tail -f` |
| `best_model.pt` | pesi al valore migliore di `--select-on` (default macro-AUC), con le metriche per classe di quell'epoca |
| `last_model.pt` | pesi + stato di optimizer/scheduler/RNG per `--resume` |
| `train.log` | copia di stdout, con `--log-file` |
| `../index.jsonl` | **una riga per run**, appesa a fine run: config, `best_score`, `best_epoch`, `stop_reason`, costo, recall per classe all'epoca selezionata. È il file che una ricerca automatica rilegge per ricordarsi la campagna, invece di riaprire ogni `experiment_record.json` (che a 100 epoche pesa ~400 KB l'uno) |

Ogni epoca stampa (e logga) due righe:

```
[epoch 001/100] lr=3.00e-04 train: loss=0.9632 acc=0.6623 | val: loss=1.1044 acc=0.6321 macroAUC=0.8288 macroF1=0.2217 balAcc=0.2890 (194.7s, eta 5h22m)  * best
            val recall/class: 0:0.03 1:0.81 2:0.36 3:0.00 4:0.00 5:0.82 6:0.00
```

La seconda riga è quella che conta su questo dataset: la recall per classe mostra
subito se il modello è collassato sulla classe maggioritaria. Nell'esempio sopra
`val_acc=0.632` sembra ragionevole, ma melanoma (4) e dermatofibroma (3) sono a
recall 0.00 — informazione invisibile guardando accuracy e loss.

Flag di monitoraggio:

| Flag | Effetto |
| --- | --- |
| `--log-file` | duplica stdout in `<out>/train.log` (o nel path passato). Le barre tqdm vanno su stderr, quindi il log resta pulito e greppabile |
| `--no-progress` | disattiva tqdm — da usare con `nohup` o in CI |
| `--resume` | riprende da `<out>/last_model.pt`: modello, optimizer, scheduler, stato RNG e `history` |
| `--index-file` | dove appendere la riga di indice (default `<out>/../index.jsonl`; `none` disattiva) |

```bash
# run lungo in background, log su file
nohup python train.py --epochs 100 --out runs/fpvit_run1 --no-progress --log-file &
tail -f runs/fpvit_run1/train.log

# ripresa dopo un'interruzione (stesso --out e stesso --epochs)
python train.py --epochs 100 --out runs/fpvit_run1 --resume
```

Attenzione al `--resume`: lo `state_dict` di `CosineAnnealingLR` contiene `T_max`,
quindi la ripresa ripristina lo **schedule originale**. Se riprendi con un
`--epochs` diverso da quello iniziale il learning rate non viene ri-disteso sul
nuovo numero di epoche (resta a `eta_min` per quelle in più); `train.py` stampa un
warning elencando le differenze di config.

Nota sullo spazio: con AdamW `last_model.pt` pesa ~200 MB (pesi + i due momenti),
contro ~68 MB di `best_model.pt`. È un singolo file sovrascritto a ogni epoca, non
una serie, quindi l'occupazione resta costante.

### Arresto anticipato e budget

Servono a una ricerca automatica: senza, ogni configurazione sbagliata costa
comunque 5.5 h, e con una ventina di run in coda il budget se ne va tutto sui
run già morti.

| Flag | Effetto |
| --- | --- |
| `--early-stop-patience N` | ferma il run dopo N epoche senza miglioramento di `--select-on` (0 = disattivato, comportamento originale) |
| `--early-stop-min-delta D` | un miglioramento inferiore a D non azzera il contatore di pazienza — evita che un run sopravviva grazie alla quinta cifra decimale |
| `--max-seconds S` | budget wall-clock **di questa invocazione**. Il loop si ferma *prima* di iniziare un'epoca che prevede di sforare (stima dalla media delle ultime 5), così il budget è un tetto e non una media. La prima epoca non viene mai bloccata: un budget più corto di un'epoca produrrebbe altrimenti un run senza alcun punteggio |

Il motivo dello stop finisce in `stop_reason` nel record e nell'indice:
`completed` \| `early_stop` \| `max_seconds` \| `interrupted`. Serve a chi
confronta i run: un trial troncato a 4 epoche e uno arrivato a convergenza non
sono confrontabili sul solo `best_score`.

`SIGTERM` è trattato come `Ctrl-C`: uno scheduler che uccide un trial poco
promettente lascia comunque record, checkpoint e riga di indice scritti.

### Cosa manca ancora

- Nessun TensorBoard / Weights & Biases: il CSV è pensato per essere plottato a
  posteriori.
- Nessuna metrica per classe sul *train* set (solo loss e accuracy aggregate).
- L'indice viene scritto **a fine run**: un `kill -9` non lascia alcuna riga
  (il `experiment_record.json`, riscritto a ogni epoca, resta la fonte di
  verità per i run interrotti a forza).

## Iperparametri principali (default = valori del paper)

| Parametro | Default | Note |
| --- | --- | --- |
| `--depth` | 4 | layer Transformer per testa ViT (paper: 4, ablation anche 6/8) |
| `--num-heads` | 3 | teste di attenzione per layer Transformer |
| `--embed-dim` | 192 | dimensione condivisa di tutti i vettori di attivazione |
| `--optimizer` | sgd | il paper usa SGD, lr=1e-3; `adamw` è un'alternativa più robusta per i layer ViT |
| `--no-resnet-head` | off | riproduce l'ablation "3 heads" |
| `--aug-preset` | `default` | vedi la sezione Augmentation |
| `--no-cutout` | off | scorciatoia: forza `cutout=false` dopo la risoluzione del preset |
| `--select-on` | `macro_auc` | metrica di validazione che seleziona `best_model.pt`: `macro_auc`, `balanced_acc`, `macro_f1`, `acc`, `loss` |
| `--class-weight` | `none` | pesi per classe nella loss di **training**: `inverse` = `N/(K·n_c)`; `effective` = numero efficace (Cui et al., CVPR 2019) |
| `--cb-beta` | 0.999 | β di `--class-weight effective`; più vicino a 1 = più vicino alla frequenza inversa |
| `--balanced-sampler` | off | ricampiona il train split perché ogni epoca sia bilanciata in media |

### Sbilanciamento: quale leva

Il train split è `[228, 359, 769, 80, 779, 4693, 99]`. Le due correzioni sono
**alternative, non complementari** — attivarle insieme corregge due volte, e
`train.py` lo segnala con un warning (resta permesso: può essere l'ablation che
si vuole).

| Schema | Rapporto peso classe 3 / classe 5 |
| --- | --- |
| `inverse` | 58.7× |
| `effective` (β=0.999) | 12.9× |

Entrambi sono riscalati a media 1 sulle classi, così l'ordine di grandezza
della loss — e quindi l'intervallo di learning rate utilizzabile — resta
confrontabile con un run non pesato. **I pesi si applicano alla sola loss di
training**: `val_loss` resta non pesata, quindi confrontabile tra run con
schemi diversi e utilizzabile come `--select-on loss`. La `train_loss`, invece,
non è confrontabile tra schemi diversi — quei run vanno confrontati sulle
metriche di validazione.

## 224 px e dataset senza leakage (DermaMNIST-C / -E)

Lo split ufficiale di DermaMNIST è fatto per immagine: la stessa lesione di
HAM10000 compare in train, val e test (Abhishek, Jain & Hamarneh, in
`legacy_first_agent/docs/references/`). `--dataset` sceglie tra:

| `--dataset` | split | train / val / test | risoluzioni |
| --- | --- | --- | --- |
| `dermamnist` (default) | ufficiale, per immagine | 7007 / 1003 / 2005 | 28 (64/128/224 via MedMNIST+, stesso split) |
| `dermamnist_c` | corretto: le immagini di lesioni presenti nel train sono spostate nel train | 8215 / 573 / 1227 | 28, 224 |
| `dermamnist_e` | train = tutto HAM10000, val/test = ISIC 2018 | 10015 / 193 / 1511 | 28, 224 |

C ed E sono scaricati da Zenodo al primo uso, con verifica MD5. A 224 px sono
ridimensionati **direttamente dagli originali** (bicubica), non ingranditi dal
28. Il protocollo previsto: train e selezione su C, test finale sul test di C
e, come test esterno, sul test di E. Il contrario non vale: il train di E
contiene tutte le immagini di val e test di C.

**FPViT a 224.** Con una patch 1×1 la prima testa ViT vedrebbe 224² = 50.176
token, quindi sopra i 64 px FPViT usa:

- lo stem ImageNet (conv 7×7 stride 2 + max-pool): B1..B4 = 56, 28, 14, 7;
- per ogni testa ViT, un token per blocco p×p (patch embedding convolutivo)
  in modo che ognuna veda una griglia 14×14 = 196 token (p = 4, 2, 1);
- con `--pretrained`, l'estrattore parte dai pesi ImageNet di ResNet-18 di
  torchvision. La corrispondenza è uno a uno: le feature sono identiche a
  quelle di torchvision (verificato). Sono pesi addestrati solo su ImageNet,
  quindi niente dati dermatologici e niente leakage. In questo caso la
  normalizzazione passa alle statistiche ImageNet.

| Parametro | Default | Note |
| --- | --- | --- |
| `--dataset` | `dermamnist` | vedi tabella sopra |
| `--img-size` | 28 | 28 o 224; le CNN restano solo a 28 |
| `--stem` | `auto` | `small` fino a 64 px, `imagenet` sopra |
| `--token-grid` | -1 (auto) | lato della griglia di token per testa; 0 = un token per posizione (il paper), auto = 0 con lo stem small, 14 con quello imagenet |
| `--pretrained` | off | estrattore ImageNet; richiede lo stem imagenet |
| `--norm` | `auto` | `imagenet` con `--pretrained`, altrimenti `dermamnist` |
| `--backbone-lr-mult` | 1.0 | lr dell'estrattore = lr × mult (es. 0.1 con `--pretrained`) |
| `--warmup-epochs` | 0 | warm-up lineare da 0.1×lr prima del coseno |
| `--amp` | off | mixed precision fp16, solo CUDA |
| `--grad-clip` | 0 | norma massima del gradiente (0 = off) |
| `--label-smoothing` | 0 | solo sulla loss di training |

Tutti i valori `auto` vengono risolti prima del run e salvati nella `config`
del checkpoint. Checkpoint precedenti senza queste chiavi si ricaricano a 28
px come prima. `predict.py`, `evaluate_test.py` e il `ModelZoo` del testing
agent ridimensionano ogni immagine alla risoluzione del modello e applicano la
sua normalizzazione (`fpvit.dataset.eval_transform_for`). `evaluate_test.py
--dataset dermamnist_e` valuta un modello di C sul test esterno. Il gate di
promozione del training agent accetta per ora solo modelli a 28 px dello split
ufficiale, perché confronta tutti sulla stessa validation.

Il notebook `colab/fpvit_224_dermamnist_c.ipynb` esegue l'ablation da zero
contro pre-addestrato (3 seed ciascuna) su GPU Colab, riprendibile dopo una
disconnessione, con il test finale su C ed E alla fine.

## Collegamento al progetto AgenticDerma

- **WP3 (Training)**: FPViT è un candidato aggiuntivo da confrontare, sotto
  lo stesso protocollo (stessi split, stessa policy di seed, stesse metriche
  di validazione), con la famiglia di riferimento ResNet-18 e con le
  alternative compatte citate nella proposta (EfficientNet-B0,
  ConvNeXt-Tiny) — soddisfa il requisito "almeno tre configurazioni di
  classificatore" (LA3.1).
- **WP4 (Evaluation)**: `fpvit/engine.evaluate()` calcola già le metriche
  richieste (accuracy, macro-F1, balanced accuracy, macro AUROC one-vs-rest,
  precision/recall/F1 per classe, matrice di confusione); manca solo la
  calibrazione, da aggiungere quando si fissa la policy di selezione finale.
- **WP2 (Data)**: `fpvit/dataset.py` supporta oltre allo split ufficiale
  DermaMNIST-C/E, le versioni corrette per lesione di Abhishek, Jain &
  Hamarneh (vedi la sezione 224 px). Il training agent e il gate di promozione
  usano ancora lo split ufficiale.
- Il checkpoint e il file `experiment_record.json` prodotti da `train.py`
  forniscono la traccia richiesta dal registro degli esperimenti (D3.3):
  configurazione completa, seed, storia delle metriche per epoca, percorso
  del checkpoint migliore.

## Limiti noti

- Il modulo `evaluate_test.py` va eseguito **una sola volta**, dopo aver
  congelato il modello selezionato: eseguirlo più volte durante la fase di
  tuning viola l'isolamento del test set richiesto da WP6/R4 della proposta.
- I valori di normalizzazione in `dataset.py` sono stati verificati sul
  train split ufficiale; se l'audit dei dati (WP2) cambia la policy di
  split, ricalcola mean/std sul nuovo set di training.
- **Sbilanciamento delle classi: gestito solo se lo si chiede.** Il train
  split è `[228, 359, 769, 80, 779, 4693, 99]`: la classe 5 (melanocytic
  nevi) è il 67 % dei campioni, quindi predire sempre 5 dà già 0.669 di
  accuracy. `--class-weight` e `--balanced-sampler` esistono ma sono **off
  per default**, perché sono deviazioni dalla ricetta del paper: vanno
  decise esplicitamente in WP3. Senza di essi il comportamento è quello di
  prima, e l'accuracy da sola è ingannevole — usa balanced accuracy e
  macro-F1 per la selezione.
- La selezione del checkpoint usa per default la macro-AUC di validazione,
  che è poco sensibile allo sbilanciamento. Nel run a 10 epoche in `runs/`
  questo è già visibile: la macro-AUC sceglie l'epoca 9, mentre balanced
  accuracy, macro-F1 e val_loss scelgono tutte l'epoca 10. Usa
  `--select-on balanced_acc` se è quella la metrica che conta.
- `--resume` ricalcola l'epoca migliore dalla `history` con il `--select-on`
  corrente, non si fida dello scalare salvato. Se si riprende cambiando
  metrica, `best_model.pt` contiene ancora i pesi dell'epoca selezionata
  prima e viene sovrascritto solo quando una nuova epoca batte il punteggio
  ricalcolato — `train.py` lo stampa come warning.
- `FPViT(multi_label=...)` è accettato e memorizzato ma non usato: la loss è
  sempre `CrossEntropyLoss`. Per un task multi-label va cambiata la loss in
  `engine.py`.
