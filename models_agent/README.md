# `models_agent/`: domande sui modelli disponibili

Risponde alle domande dell'utente sui modelli del sistema e sui loro dati:
quali modelli ci sono, come sono stati addestrati, accuracy, macro-F1, AUC e
curve ROC, perché un run è escluso, com'è composto il set di validazione
(DermaMNIST-C val), come funzionano il voto e il gate di promozione.

È **di sola lettura**: nessuna interruzione per l'umano, nessun file scritto,
nessun modello toccato. L'orchestrator lo chiama quando il router classifica
una richiesta testuale come `models`, oppure quando l'utente sceglie "Show the
available models" nel dialogo di chiarimento. Nella piattaforma non prende
nessun lock, quindi non aspetta la fine di una campagna di training.

```
START -> collect_facts -> explain -> check_claims -> END
```

| nodo | chi | cosa fa |
| --- | --- | --- |
| `collect_facts` | codice (`facts.py`) | Raccoglie i fatti dei membri dell'ensemble e dei run esclusi: architettura, risoluzione, dati di training, pre-addestramento ImageNet, parametri, ottimizzatore, epoche, peso nel voto, classi mai riconosciute, promozione (dal registro) ed esclusione (da `ensemble_exclusions.json`). Calcola le metriche su **DermaMNIST-C val** dalle probabilità già in cache in `val_dermamnist_c.json`, senza nuova inferenza: accuracy, balanced accuracy, macro-F1, macro-AUC, precision/recall/F1/AUC per classe. Stesse metriche per l'ensemble (voto soft pesato), più le curve ROC one-vs-rest e le matrici di confusione. Aggiunge la composizione dei dataset (conteggi per classe dei train e val) e le regole del sistema. |
| `explain` | LLM | Risponde alla domanda usando **solo** i fatti. Senza LLM la risposta è il riassunto deterministico (`facts.summary`). |
| `check_claims` | codice | Ogni numero e ogni nome con underscore (run, dataset, metriche) citato nella risposta deve comparire nei fatti o nella domanda. I numeri si confrontano alla precisione con cui sono scritti (0.76 corrisponde a 0.756, 75.6 % anche). Se qualcosa non torna, l'utente vede il riassunto deterministico e l'elenco di ciò che è stato respinto. |

## Cosa l'agente non legge mai

Il **test split**, nemmeno le etichette: i conteggi per classe vengono solo
dai train e dalle validation. Tutte le metriche sono di validation, e i fatti
lo dicono. Anche le motivazioni in `ensemble_exclusions.json` citano solo
numeri di validation, perché finiscono nel contesto dell'LLM.

## Cosa vede la piattaforma

- **Risposta** dell'LLM (o riassunto), con chi l'ha scritta e l'esito del controllo.
- **Tabella** di caratteristiche e metriche: membri, ensemble, esclusi.
- **Recall per classe**, con il numero di immagini di ogni classe in C-val.
- **Curve ROC per classe**: una linea per modello, l'ensemble in nero, gli esclusi tratteggiati in grigio. Passando sopra una curva si vedono modello e AUC.
- **Matrice di confusione** dell'ensemble, normalizzata per riga.
- **Avvertenze**: C-val è piccolo (5 dermatofibromi, 7 lesioni vascolari), le metriche dell'ensemble sono ottimistiche, i modelli dello split ufficiale hanno scelto l'epoca su una validation che contiene C-val.

Il pulsante "Reasoning context" apre ciò che l'LLM riceve esattamente: il prompt
di sistema e i fatti, circa 13.700 caratteri.

## Limiti noti

- Il controllo verifica che ogni numero esista nei fatti, **non** che sia
  attribuito al modello giusto: un valore vero associato al modello sbagliato
  passa. I calcoli nuovi (differenze, medie) e i numeri inventati vengono
  invece respinti.
- Con un LLM locale piccolo (qwen3.6 su Ollama) una risposta richiede circa due minuti.
- I nomi senza underscore (per esempio `probe`) non vengono controllati.

## Verificato

- Router con qwen3.6 su 7 frasi di prova: le 4 domande sui modelli vanno a
  `models`, le 2 richieste di addestramento a `train`, il saluto a `unclear`.
- Risposte di qwen3.6 a "Da cosa è composto il C-val?" e "Quale modello è il
  migliore per il melanoma, e quanto è affidabile quel numero?": corrette e
  passate dal controllo.
- Il controllo respinge una differenza calcolata (0.168), un modello inventato
  e un numero del test (0.783).
- Piattaforma: domanda con LLM (instradata, risposta verificata, tabelle, ROC,
  matrice); domanda senza LLM tramite il dialogo di chiarimento; previsione e
  campagna di training invariate.
