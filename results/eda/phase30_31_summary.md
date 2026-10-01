# Phase 30–31 — Peso 2 e profondità 7 contro Phase 14

Entrambe le prove sono concluse e verificate. Nessuna soddisfa il criterio operativo del piano: F1 aggregato superiore a Phase 14 e miglioramento in almeno due fold su tre. Conservare Phase 14, peso 1 e profondità 5.

262.508 righe development e 295 feature, inclusi ALCDAY5 e HAREHAB1. Stessi split e seed della baseline; tre fold esterni e due interni per scegliere ogni soglia. Nove nuovi fit per candidato. Baseline storica riutilizzata dopo verifica numerica esatta.

Phase 30 modifica solo il peso dei positivi da 1 a 2, mantenendo profondità 5. La loss logistica, gradienti, Hessiani e prevalenza iniziale usano i pesi; quantili e minimo di 200 righe per foglia restano basati sulle righe originali. Le metriche e la selezione F1 sono non pesate.

Phase 31 modifica solo la profondità massima da 5 a 7 e mantiene peso 1. Entrambe usano 200 alberi, learning rate 0,05, L2=1, 64 bin e 128 feature candidate per nodo. Non è stata provata la combinazione peso 2 più profondità 7.

| Metrica | Phase 14 | Phase 30, peso 2 | Phase 31, profondità 7 |
| --- | ---: | ---: | ---: |
| f1 | 0.442155227 | 0.439118448 | 0.441160950 |
| precision | 0.367632339 | 0.361155162 | 0.364346970 |
| recall | 0.554572908 | 0.560008628 | 0.559016393 |
| average_precision | 0.428609250 | 0.428237932 | 0.428748338 |
| log_loss | 0.215225071 | 0.231722615 | 0.215090233 |

| Fold | F1 Phase 14 | F1 peso 2 | Delta peso 2 | F1 profondità 7 | Delta profondità 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.439026847 | 0.433710992 | -0.005315855 | 0.436892735 | -0.002134113 |
| 2 | 0.444053521 | 0.440990968 | -0.003062553 | 0.444066355 | +0.000012834 |
| 3 | 0.443511646 | 0.442857143 | -0.000654503 | 0.442909091 | -0.000602555 |

| Fold | Soglia Phase 14 | Soglia peso 2 | Soglia profondità 7 |
| --- | ---: | ---: | ---: |
| 1 | 0.203056689 | 0.331399741 | 0.197111771 |
| 2 | 0.220104637 | 0.338300890 | 0.220880536 |
| 3 | 0.209445973 | 0.351041141 | 0.215307776 |

Le soglie sono scelte soltanto sugli OOF interni, mai sulle righe esterne poi valutate. La soglia maggiore del modello pesato riflette una scala degli score diversa: non è di per sé un segnale di migliore o peggiore classificazione. Non sono nuove soglie di submission.

| Esito | Phase 14 | Peso 2 | Profondità 7 |
| --- | ---: | ---: | ---: |
| true_positives | 12855 | 12981 | 12958 |
| false_negatives | 10325 | 10199 | 10222 |
| false_positives | 22112 | 22962 | 22607 |
| true_negatives | 217216 | 216366 | 216721 |

Peso 2: F1 0,439118448, delta −0,003036779, tutti e tre i fold inferiori. Recupera 126 positivi ma aggiunge 850 falsi positivi. Il recall sale da 0,554573 a 0,560009; la precision scende da 0,367632 a 0,361155. AP diminuisce leggermente (−0,000371); log loss non pesata aumenta a 0,231723 (+0,016498). La loss pesata modifica la scala probabilistica verso i positivi; gli score grezzi non vanno trattati come probabilità già calibrate sulla prevalenza originale. Questo limite della log loss va distinto dal peggioramento F1, che persiste dopo nuova selezione interna delle soglie.

Profondità 7: F1 0,441160950, delta −0,000994277. Il fold 2 migliora soltanto di +0,000012834; gli altri due peggiorano. Recupera 103 positivi e aggiunge 495 falsi positivi. Recall 0,559016, precision 0,364347. AP aumenta di appena 0,000139 e log loss diminuisce di 0,000135; sono guadagni piccoli su metriche secondarie e non giustificano il modello più profondo per l’obiettivo F1.

Intervallo bootstrap descrittivo del delta F1 Phase 30: [-0.004730150; -0.001318961].
Intervallo bootstrap descrittivo del delta F1 Phase 31: [-0.002978237; +0.001004398].

20.000 ripetizioni appaiate, seed 20260929, fit e soglie fissi. Il delta di peso 2 è coerentemente negativo nei fold e nell’intervallo condizionale. L’intervallo della profondità 7 comprende zero: il calo osservato è piccolo e incerto, ma manca un vantaggio coerente. Non sono incluse la variabilità da riaddestramento né l’incertezza della selezione ripetuta sui fold development. Non si può inferire che ogni altro peso o profondità debba peggiorare.

CV peso 2: 48.55 minuti. CV profondità 7: 63.89 minuti. Durata complessiva del coordinatore, incluse le verifiche finali: 64.17 minuti. Avvio simultaneo alle 01:11 del 1 ottobre 2026; peso 2 concluso alle 02:00 e profondità 7 alle 02:15, fuso Europe/Rome. I tempi CV escludono preflight e verifica finale; risorse condivise durante la sovrapposizione, senza confronto misurato con esecuzione sequenziale.

144 test prima del training, inclusi sei nuovi test della pesatura; predizioni e score del caso peso 1 identici alla versione precedente. Tre checkpoint storici Phase 14 riprodotti esattamente per ciascun preflight. Tutte le nove partizioni conservano feature, ordine, righe e binning della baseline. Sei checkpoint esterni dei nuovi candidati riproducono esattamente gli OOF; pesi, profondità, prevalenze iniziali di training, soglie e metriche verificati. Entrambi i processi terminati con codice 0. Record pubblicati in sequenza nel registro condiviso e sorgenti eseguiti conservati.

Decisione: entrambe le configurazioni concluse e non adottate; mantenere Phase 14, peso 1, profondità 5, 200 alberi e learning rate 0,05. Questi confronti non hanno avviato combinazioni o prodotto una nuova submission. I successivi confronti MLP sono conclusi nella [Phase 32](phase32_mlp_summary.md). HAREHAB1 è conservata nella versione finale; la verifica nell'ambiente ufficiale resta da eseguire. La [verifica finale](../../docs/FINAL_AUDIT.md) riporta lo stato della consegna.

- [Record Phase 30](../experiments/20261001T000013003849Z_phase30-phase14-positive-weight2-depth5-features128-200trees.json).
- [Confronto Phase 30](../experiments/phase30_comparison.json).
- [Record Phase 31](../experiments/20261001T001533065665Z_phase31-phase14-depth7-depth7-features128-200trees.json).
- [Confronto Phase 31](../experiments/phase31_comparison.json).
