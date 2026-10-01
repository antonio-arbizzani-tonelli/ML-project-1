# Phase 19: trasferimento del tuning sulla Phase 14

## Verifica e protocollo

La verifica di conformità ha confrontato dati, etichette, split, configurazioni, percorso del preprocessing e predizioni dei checkpoint Phase 14. Tutti i 35 controlli sono passati; le predizioni ricalcolate coincidono esattamente nei tre fold. Nessun modello è stato addestrato durante questa verifica.

Le due prove conservano la rappresentazione Phase 14 con `HAREHAB1`, 295 feature e tutti i seed originali. Cambiano un solo parametro. Ciascuna usa tre fold esterni e due fold interni per scegliere la soglia: nove training per candidato, sui soli 262.508 esempi development.

## Metriche annidate aggregate

| Modello | F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 14 | 0.442155 | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Foglia 150 | 0.439991 | 0.361279 | 0.562554 | 0.429144 | 0.215229 |
| Learning rate 0,07 | 0.440700 | 0.368610 | 0.547843 | 0.429091 | 0.215133 |

## F1 per fold

| Modello | Fold 1 | Fold 2 | Fold 3 | Media dei fold |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 | 0.439027 | 0.444054 | 0.443512 | 0.442197 |
| Foglia 150 | 0.435083 | 0.441410 | 0.443472 | 0.439988 |
| Learning rate 0,07 | 0.437423 | 0.442220 | 0.442727 | 0.440790 |

## Soglie scelte nei fold interni

| Modello | Fold 1 | Fold 2 | Fold 3 | Media |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 | 0.203057 | 0.220105 | 0.209446 | 0.210869 |
| Foglia 150 | 0.205464 | 0.203839 | 0.206650 | 0.205318 |
| Learning rate 0,07 | 0.198371 | 0.219143 | 0.221450 | 0.212988 |

## Matrici di confusione

| Modello | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 | 12855 | 10325 | 22112 | 217216 |
| Foglia 150 | 13040 | 10140 | 23054 | 216274 |
| Learning rate 0,07 | 12699 | 10481 | 21752 | 217576 |

## Foglia 150

Differenza F1: -0.002165; fold favorevoli: 0/3. Tempo della validazione: 59.20 minuti.

Rispetto al riferimento: +185 veri positivi e +942 falsi positivi. La variazione dei falsi negativi è -185.

Fra le righe che cambiano decisione: 483 falsi negativi vengono recuperati, 298 veri positivi vengono persi, 1926 nuovi falsi positivi compaiono e 984 falsi positivi vengono eliminati.

Le differenze F1 dei tre fold sono -0.003944, -0.002644 e -0.000040. AP cambia di +0.000535; log loss cambia di +0.000004.

Bootstrap appaiato descrittivo: intervallo 95% della differenza F1 [-0.003801, -0.000494]. Mantiene fissi modelli e soglie e non misura l'incertezza da riaddestramento o selezione ripetuta dei candidati.

Record: [20260929T224222134395Z_phase19-phase14-leaf150-depth5-features128-200trees.json](../experiments/20260929T224222134395Z_phase19-phase14-leaf150-depth5-features128-200trees.json).

## Learning rate 0,07

Differenza F1: -0.001455; fold favorevoli: 0/3. Tempo della validazione: 59.45 minuti.

Rispetto al riferimento: -156 veri positivi e -360 falsi positivi. La variazione dei falsi negativi è +156.

Fra le righe che cambiano decisione: 393 falsi negativi vengono recuperati, 549 veri positivi vengono persi, 1590 nuovi falsi positivi compaiono e 1950 falsi positivi vengono eliminati.

Le differenze F1 dei tre fold sono -0.001604, -0.001834 e -0.000785. AP cambia di +0.000481; log loss cambia di -0.000092.

Bootstrap appaiato descrittivo: intervallo 95% della differenza F1 [-0.003297, +0.000415]. Mantiene fissi modelli e soglie e non misura l'incertezza da riaddestramento o selezione ripetuta dei candidati.

Record: [20260929T224236985386Z_phase19-phase14-lr07-depth5-features128-200trees.json](../experiments/20260929T224236985386Z_phase19-phase14-lr07-depth5-features128-200trees.json).

## Decisione

La prova misura il trasferimento alla rappresentazione Phase 14. L'esito diverso dalla Phase 17 non può essere attribuito alla sola HAREHAB1: fra i due rami differiscono anche altre scelte di preprocessing.

Nessun candidato soddisfa la regola del piano: F1 aggregato superiore al controllo e miglioramento in almeno due fold su tre. Conservare Phase 14 con minimo foglia 200 e learning rate 0.05. Il piccolo aumento dell'AP non sostituisce un miglioramento dell'F1, che resta la metrica principale.

Il vantaggio della Phase 17 sul suo ramo senza HAREHAB1 non si trasferisce al ramo Phase 14 nelle due prove isolate. Non assumere un effetto additivo e non avviare una combinazione come se i due cambiamenti fossero già favorevoli. L'ablazione della sola ALCDAY5, conservando DROCDY3_ e tutti gli altri input del riferimento, è stata successivamente completata nella [Phase 28](phase28_phase14_ablate_alcday5_summary.md) e non è stata adottata.

## Ambiente e artefatti

I due candidati sono stati eseguiti in parallelo. I tempi misurano la durata di ciascuna validazione in queste condizioni; il tempo storico Phase 14 non è un confronto controllato di velocità.

Ambiente dei record: Python 3.14.3 e NumPy 2.4.2. I test locali passano (86/86) e le predizioni dei checkpoint di riferimento sono riprodotte esattamente. Questo non sostituisce la verifica nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1.

## Precisazione sulla Phase 16 remota

La configurazione remota contiene cinque interazioni, ma il preprocessing ad alberi versionato non legge il campo interactions. Le colonne sorgente conservate e l'output dichiarato nel record sono entrambi 295. Il risultato non dimostra una prova effettiva di tali interazioni. Il piano e la storia degli esperimenti sono stati corretti; questa precisazione non cambia le due prove Phase 19 sulla rappresentazione Phase 14 verificata.

I JSON degli esperimenti e il confronto completo sono versionabili. Checkpoint, log e predizioni OOF restano negli artefatti locali. La configurazione finale non viene modificata da questo script.
