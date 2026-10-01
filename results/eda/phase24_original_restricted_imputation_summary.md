# Phase 24 — conferma annidata dell'imputazione ristretta originale

## Configurazione e protocollo

Replica della pipeline Phase 16 originale: 303 input, otto indicatori con binning a quantili, imputazione ristretta di INCOME2, PNEUVAC3, EMPLOY1, altezza/peso e BMI derivabile. I due indicatori rari restano accorpati dal binning originale. Non è la variante con binning esatto della Phase 23.

Stessi dati development e tre fold esterni di Phase 14. Booster: 200 alberi, profondità 5, learning rate 0,05, minimo foglia 200, 64 bin, 128 feature candidate, L2=1, seed 20260920. Soglie da due fold interni con seed 20260922 + fold esterno. Imputatori rifatti esclusivamente sul training della rispettiva partizione, senza target di malattia.

## Verifiche e riutilizzo

Riutilizzati i tre modelli esterni originali dopo confronto di hash, righe e label di training, parametri e bin. Le predizioni esterne e gli score di training ricalcolati coincidono esattamente con quelli storici. Addestrati sei nuovi modelli interni; salvate le loro predizioni con righe, fold e hash. Soglie e risultati ricostruiti dagli artefatti.

Tempo della nuova CV interna, inclusi imputatori: 22.24 minuti. Il tempo non comprende i tre training esterni storici né il preflight. Ambiente: Python 3.14.3, NumPy 2.4.2.

## Risultati

| Metrica | Phase 14 | Imputazione originale | Differenza |
| --- | ---: | ---: | ---: |
| f1 | 0.442155227 | 0.441790736 | -0.000364491 |
| precision | 0.367632339 | 0.367812464 | +0.000180125 |
| recall | 0.554572908 | 0.553019845 | -0.001553063 |
| average_precision | 0.428609250 | 0.428610705 | +0.000001455 |
| log_loss | 0.215225071 | 0.215230306 | +0.000005235 |

| Fold | F1 Phase 14 | F1 imputazione | Differenza | Soglia interna |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.439026847 | 0.437723315 | -0.001303532 | 0.203563099 |
| 2 | 0.444053521 | 0.442716815 | -0.001336706 | 0.215518614 |
| 3 | 0.443511646 | 0.445095168 | +0.001583523 | 0.213873795 |

| Esito | Phase 14 | Imputazione originale | Differenza |
| --- | ---: | ---: | ---: |
| Veri positivi | 12.855 | 12.819 | −36 |
| Falsi negativi | 10.325 | 10.361 | +36 |
| Falsi positivi | 22.112 | 22.033 | −79 |
| Veri negativi | 217.216 | 217.295 | +79 |

## Interpretazione e decisione

Il precedente F1 esplorativo 0.442673703 sceglieva e valutava la soglia sulle stesse label OOF. L'F1 annidato qui riportato separa questa scelta dalla valutazione esterna; il riutilizzo di predizioni già consultate e la selezione ripetuta dei candidati mantengono un limite di selezione del modello.

Fold favorevoli: **1/3**. Il candidato non soddisfa la regola operativa del piano: F1 aggregato superiore al controllo e miglioramento in almeno due fold. **Imputazione originale non adottata; Phase 14 e la sua soglia finale restano configurate.** AP e log loss sono praticamente identici al controllo. Il piccolo vantaggio esplorativo non è confermato dalla selezione annidata delle soglie.

Intervallo descrittivo bootstrap appaiato 95% del delta F1: **[−0,002125; +0,001351]**, 20.000 ripetizioni, seed 20260929. Comprende zero: il risultato non dimostra una perdita certa, ma non conferma un miglioramento utile. Modelli e soglie fissi; non comprende incertezza da riaddestramento o selezione dei candidati.

Record: [20260930T114012414713Z_phase24-original-restricted-imputation-depth5-features128-200trees.json](../experiments/20260930T114012414713Z_phase24-original-restricted-imputation-depth5-features128-200trees.json).
Confronto completo: [JSON](../experiments/phase24_original_restricted_imputation_comparison.json).

## Stato dei confronti successivi

La verifica della configurazione originale è conclusa. La combinazione con tutti gli otto indicatori preservati tramite binning esatto resta non eseguita e viene rinviata. Phase 24 e Phase 23 differiscono anche nel binning: il loro confronto non misura il solo effetto dell'imputazione.

Il confronto con L2 delle foglie da 1 a 5 sulla Phase 14 è stato successivamente completato nella [Phase 25](phase25_l2_summary.md) e non è stato adottato.

## Test e tracciabilità

109 test superati prima del training; 110 dopo una correzione del conteggio diagnostico dei bin occupati. La correzione riguarda solo il rapporto: i due indicatori accorpati sono EMPLOY1__was_refused e PNEUVAC3__was_refused. Sono state ripetute le verifiche dei tre checkpoint, con differenze ancora pari a zero; nessun ulteriore modello di malattia addestrato. Nel record sono conservati i riferimenti e gli hash dell'helper eseguito e del preflight originale, insieme ai controlli diagnostici aggiornati. Le sorgenti numeriche, le predizioni, le soglie e le metriche sono rimaste invariate.

Sono salvati configurazione, record, confronto, OOF esterni e interni e rapporto; i tre checkpoint originali e tutti i risultati precedenti sono conservati. Modifiche locali, senza commit o push in questa attività. La compatibilità con Python 3.9 / NumPy 1.23.1 rimane da verificare prima della consegna.
