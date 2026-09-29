# FPViT a 224 px: estrattore, teste, pretrained e scratch

Questa nota spiega com'è fatto FPViT (Feature Pyramid Vision Transformer) a 224 px su DermaMNIST-C. Copre la differenza tra **estrattore** e **teste**, perché usano learning rate diversi e cosa distingue le due configurazioni dell'ablation, **`pretrained`** e **`scratch`**. I dettagli di implementazione e tutti i flag sono in [`fpvit/README.md`](fpvit/README.md). Il notebook che esegue gli esperimenti è [`colab/fpvit_224_dermamnist_c.ipynb`](colab/fpvit_224_dermamnist_c.ipynb).

## 1. Com'è fatto FPViT a 224 px

```
immagine 3×224×224
   │
   ▼  ESTRATTORE (ResNet-18)                        11,18 M parametri (65%)
 stem  conv 7×7/2 + max-pool     → 64 × 56×56             0,01 M
 layer1                          → B1: 64  × 56×56        0,15 M
 layer2                          → B2: 128 × 28×28        0,53 M
 layer3                          → B3: 256 × 14×14        2,10 M
 layer4                          → B4: 512 ×  7×7         8,39 M
   │
   ▼  TESTE                                          5,90 M parametri (35%)
 vit1 su B1: patch 4×4 → 196 token → 4 layer Transformer → vettore 192   2,01 M
 vit2 su B2: patch 2×2 → 196 token → 4 layer Transformer → vettore 192   1,92 M
 vit3 su B3: patch 1×1 → 196 token → 4 layer Transformer → vettore 192   1,87 M
 testa ResNet su B4: media globale + lineare             → vettore 192   0,10 M
   │
 concatenazione (4 × 192 = 768) → classificatore lineare → 7 classi      0,005 M

 Totale: 17,08 M parametri
```

### L'estrattore

L'estrattore trasforma i pixel in **mappe di feature a quattro scale** (B1…B4).

- Nei primi layer (B1, 56×56) ogni posizione "vede" una piccola zona dell'immagine: bordi, texture fini, reticolo pigmentato.
- Negli ultimi (B4, 7×7) ogni posizione riassume una zona ampia: forma e struttura globale della lesione.

È una ResNet-18 standard con lo stem ImageNet (conv 7×7 con stride 2 e max-pool). Per questo può caricare i pesi ImageNet di torchvision uno a uno, con feature identiche a quelle di torchvision (verificato).

Nel codice l'estrattore è `model.backbone` ([`fpvit/model.py`](fpvit/model.py), classe `ResNet18Extractor`).

### Le teste

Le teste leggono le mappe dell'estrattore e **decidono**.

- **Tre teste ViT** (`vit1`, `vit2`, `vit3`), una per scala tra B1 e B3. Ognuna divide la mappa in blocchi (4×4, 2×2, 1×1) così da ottenere sempre una griglia di 14×14 = **196 token**. Poi, con l'attenzione di 4 layer Transformer, mette in relazione zone diverse della lesione (per esempio l'asimmetria tra due lati) e la riassume in un vettore di 192 numeri (il class token).
- **Una testa ResNet** su B4: media su tutte le posizioni e un layer lineare, che produce un altro vettore di 192 numeri.
- **Un classificatore lineare** concatena i quattro vettori (768 numeri) e produce i punteggi delle 7 classi.

Questa è l'idea del paper: una "piramide" di feature a più scale, ognuna con la propria testa, fuse alla fine.

**Perché i blocchi e non un token per pixel.** A 28 px il paper usa un token per ogni posizione della mappa. A 224 px la prima testa ne vedrebbe 56×56 = 3.136, e il costo dell'attenzione cresce con il quadrato del numero di token. Con la griglia fissa di 196 token ogni testa ha la stessa lunghezza di sequenza di un ViT-B/16 a 224 px.

## 2. Pretrained e scratch

Le due configurazioni hanno **la stessa architettura** (quella sopra, 17,08 M parametri). Cambia **da dove parte l'estrattore**, più alcune impostazioni di training che ne conseguono.

### L'inizializzazione

- **`scratch`**: tutta la rete parte da pesi casuali, come nel paper di FPViT. L'estrattore deve imparare da zero anche le feature di base (bordi, texture, colore), usando solo le 8.215 immagini di training di DermaMNIST-C.
- **`pretrained`**: l'estrattore parte dai pesi di una ResNet-18 addestrata su ImageNet (circa 1,2 milioni di foto di oggetti comuni). Sa già riconoscere bordi, texture e forme, e deve solo adattarsi alla dermoscopia. **Le teste partono comunque da pesi casuali**, perché su ImageNet non esistono.

ImageNet non contiene immagini dermatologiche, quindi il pre-training **non introduce leakage** verso la validation e il test di DermaMNIST-C/E.

### Le impostazioni che cambiano di conseguenza

| | `scratch` | `pretrained` | perché |
| --- | --- | --- | --- |
| estrattore | pesi casuali | pesi ImageNet | è la differenza che l'ablation misura |
| normalizzazione | statistiche DermaMNIST | statistiche ImageNet | i pesi ImageNet si aspettano input normalizzati come in ImageNet |
| lr teste | 1e-3 | 3e-4 | con feature già buone basta un passo più piccolo |
| lr estrattore | 1e-3 (uguale alle teste) | 3e-5 (×0.1) | lo si rifinisce piano, senza cancellare le feature ImageNet |
| warmup | 5 epoche | 3 epoche | all'inizio le teste casuali producono gradienti grandi: il warmup li contiene |
| epoche max / patience | 100 / 20 | 60 / 15 | partendo da zero serve più tempo per convergere |

Sono uguali per entrambe: AdamW con weight decay 0.05, batch 64, mixed precision, gradient clipping 1.0, class weighting `effective`, augmentation `default` e selezione del checkpoint sulla balanced accuracy di validation. Ogni configurazione gira con 3 seed (42, 43, 44).

### Cosa aspettarsi

- **`pretrained`** di solito converge in meno epoche e ottiene risultati migliori quando i dati sono pochi, come qui.
- **`scratch`** è il riferimento fedele al paper. Senza di lui non si potrebbe dire quanto del guadagno viene dal pre-training.

### Un limite del confronto

Le due configurazioni non differiscono **solo** per l'inizializzazione: cambiano anche learning rate, warmup, numero di epoche e normalizzazione. L'ablation risponde quindi alla domanda *"FPViT da zero contro FPViT pre-addestrato, ognuno con una ricetta sensata per il suo caso"*, non a *"quanto vale il pre-training a parità di tutto il resto"*. Per isolare l'effetto dell'inizializzazione si può aggiungere una terza configurazione: pesi ImageNet ma con la stessa ricetta di `scratch`.

## 3. Perché teste ed estrattore hanno learning rate diversi (solo `pretrained`)

Con `pretrained` le due metà del modello partono da condizioni molto diverse:

- **le teste devono imparare tutto**, quindi usano l'lr pieno (3e-4);
- **l'estrattore va solo adattato**, dalle foto di oggetti comuni alla dermoscopia, quindi usa un lr 10 volte più piccolo (3e-5, `--backbone-lr-mult 0.1`).

C'è anche un motivo di **protezione**. Nelle prime epoche le teste casuali sbagliano molto e producono gradienti grandi, che risalgono fino all'estrattore. Con un lr alto questi gradienti "rumorosi" rovinerebbero le feature ImageNet prima che le teste abbiano imparato qualcosa di sensato, e il vantaggio del pre-training andrebbe perso: è il cosiddetto *catastrophic forgetting*. L'lr ridotto dell'estrattore e il warmup servono a evitarlo.

Con `scratch` la distinzione non serve: tutta la rete parte da zero e usa lo stesso lr.

### Dove sta nel codice

`build_optimizer` in [`train.py`](train.py) crea due gruppi di parametri quando `--backbone-lr-mult` è diverso da 1:

- **gruppo 0**: tutto tranne `model.backbone` (teste ViT, testa ResNet, classificatore), a `--lr`;
- **gruppo 1**: `model.backbone` (l'estrattore), a `--lr × --backbone-lr-mult`.

Warmup e coseno si applicano a entrambi i gruppi allo stesso modo, quindi il rapporto 10:1 resta costante per tutto il training. Anche il weight decay è uguale per entrambi.

### Il grafico "lr (teste)" del notebook

Il terzo pannello della cella 7 mostra l'lr del **gruppo 0**, cioè delle teste: è quello registrato in `metrics.csv`. L'lr dell'estrattore ha esattamente la stessa forma, scalata di 10 volte, quindi non serve un secondo grafico. Con `scratch` c'è un solo lr e il grafico mostra quello.

La curva mostra la schedule:

1. **warmup**: nelle prime epoche l'lr sale in linea retta da 0,1× al valore pieno;
2. **coseno**: poi scende dolcemente fino a quasi zero, arrivandoci all'ultima epoca prevista (60 o 100).

Se la curva si interrompe mentre l'lr è ancora alto, il run si è fermato presto (per early stop o interruzione), e il modello non ha avuto la fase finale a lr basso, in cui di solito si stabilizza. Il valore registrato per un'epoca è l'lr **con cui quell'epoca è stata allenata**.

## 4. Varianti possibili

- **Estrattore congelato** (`--backbone-lr-mult 0`): si allenano solo le teste. I pesi dell'estrattore restano fermi, mentre le statistiche di BatchNorm continuano ad aggiornarsi; per un congelamento completo servirebbe anche metterlo in modalità eval, cosa che oggi non è implementata. È veloce e non rischia di rovinare le feature, ma non adatta l'estrattore alla dermoscopia. Di solito è peggiore del fine-tuning quando le immagini sono molto diverse da ImageNet.
- **Fine-tuning a lr uniforme** (`--backbone-lr-mult 1`): utile per l'ablation "stessa ricetta di `scratch`, pesi ImageNet".
- **lr diverso per ogni layer** (*layer-wise lr decay*): lr sempre più piccolo andando verso lo stem, perché i primi layer (bordi, texture) sono i più generici e cambiano meno. È più fine, ma aggiunge un iperparametro. Oggi non è implementato.
