# Phase 23 — soli otto indicatori dei mancanti

## Cosa è stato provato

Phase 14 più gli otto indicatori della Phase 16 ristretta: non so/rifiuto di INCOME2 e PNEUVAC3, rifiuto di EMPLOY1 e mancanza di HTM4, WTKG3 e _BMI5. Nessun valore è imputato. Le 295 colonne originali e i loro quantili sono invariati; soltanto gli otto indicatori usano binning esatto. Totale: 303 input.

Stessi dati development (262.508 righe), seed e parametri Phase 14: 200 alberi, profondità 5, learning rate 0,05, minimo foglia 200, L2=1, 64 bin e 128 feature candidate per nodo. HAREHAB1 resta inclusa. Tre fold esterni e due interni: nove fit. Ogni soglia è scelta soltanto sugli OOF interni e valutata sul fold esterno.

Il controllo riutilizza gli artefatti Phase 14 verificati e non viene riaddestrato. L'audit verifica tutte le nove partizioni, l'identità delle colonne sorgente e i bin degli indicatori. Dopo il training si verificano i tre checkpoint e si ricalcolano le soglie dagli OOF interni ora salvati.

## Risultati annidati

| Modello | F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 14 | 0.442155 | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Phase 23: soli indicatori | 0.440322 | 0.363606 | 0.558067 | 0.428833 | 0.215258 |

## Risultati per fold

| Fold | F1 Phase 14 | F1 indicatori | Delta F1 | Soglia indicatori |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.439027 | 0.436573 | -0.002454 | 0.211290 |
| 2 | 0.444054 | 0.441280 | -0.002774 | 0.199453 |
| 3 | 0.443512 | 0.443071 | -0.000441 | 0.212347 |

## Matrice di confusione

| Modello | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 | 12855 | 10325 | 22112 | 217216 |
| Phase 23 | 12936 | 10244 | 22641 | 216687 |

## Interpretazione e decisione

Delta F1: -0.001833; fold favorevoli: 0/3. Tempo della CV: 72.0 minuti.

Variazione netta: +81 veri positivi e +529 falsi positivi. L'aumento di recall non compensa la perdita di precision.

Recuperati 517 falsi negativi e persi 436 veri positivi; aggiunti 1924 falsi positivi ed eliminati 1395.

AP cambia di +0.000224; log loss cambia di +0.000033. La priorità resta l'F1. Anche l'F1 esplorativo con una soglia globale ottimizzata sugli stessi OOF (0.441249) resta sotto il controllo calcolato nello stesso modo (0,442078).

Intervallo bootstrap condizionato 95% per delta F1: [-0.003648, -0.000013], a modelli e soglie fissi; non include variabilità del training o selezione ripetuta.

Il candidato non soddisfa il filtro operativo del piano. Phase 14 resta il riferimento; il risultato non esclude un beneficio quando agli indicatori si aggiunge l'imputazione.

La successiva prova con imputazione ristretta resta da eseguire, anche se gli indicatori soli perdono. Il precedente F1 esplorativo 0,442674 riguardava indicatori e imputazione insieme, con una soglia scelta sugli stessi OOF valutati e due indicatori annullati dal binning: questa prova non è la conferma di quella combinazione.

## Frequenze degli indicatori nei training esterni

| Indicatore | Minimo % | Massimo % |
| --- | ---: | ---: |
| EMPLOY1__was_refused | 0.843 | 0.867 |
| INCOME2__was_unknown | 7.857 | 7.947 |
| INCOME2__was_refused | 9.224 | 9.343 |
| PNEUVAC3__was_unknown | 7.641 | 7.691 |
| PNEUVAC3__was_refused | 0.455 | 0.489 |
| HTM4__was_blank | 3.451 | 3.492 |
| WTKG3__was_blank | 6.957 | 7.023 |
| _BMI5__was_blank | 8.227 | 8.290 |

La frequenza descrive gli indicatori e non misura la loro utilità predittiva. Aggiungere otto colonne modifica anche il campionamento delle feature, pur conservando 128 candidate per nodo: il risultato riguarda la pipeline completa.

## Artefatti e limiti

Prima del training sono passati 104 test, inclusi conservazione degli OOF interni, confini dei fold e identità dei risultati con/senza salvataggio. Dopo il training le tre predizioni dei checkpoint e le tre soglie interne coincidono esattamente con gli artefatti registrati.

Ambiente: Python 3.14.3, NumPy 2.4.2; ambiente ufficiale Python 3.9 / NumPy 1.23.1 ancora da verificare.

Il 20% iniziale non è utilizzato nella prova, ma era già stato consultato storicamente. Configurazione finale e soglia di submission restano invariate; nessun nuovo modello sull'intero training o submission.

- [Record completo](../experiments/20260930T103147105876Z_phase23-phase14-missing-reason-flags-depth5-features128-200trees.json).
- [Confronto verificato](../experiments/phase23_missing_reason_flags_comparison.json).
- [Audit prima del training](phase23_missing_reason_flags_preflight.json).
- Predizioni esterne, tre checkpoint e tre artefatti OOF interni in results/boosting_artifacts/phase23_phase14_missing_reason_flags/.
