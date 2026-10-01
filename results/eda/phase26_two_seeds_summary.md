# Phase 26 — Secondo seed e media uniforme dei due booster

## Protocollo

Phase 14 invariata: 295 feature inclusa HAREHAB1, preprocessing codebook, binning a quantili, NaN gestiti dagli alberi; 200 alberi, profondità 5, learning rate 0,05, foglia minima 200, 64 bin, 128 feature candidate, L2=1.

Seed 20260920 e 20260921; pesi fissati a 0,5 prima della valutazione. Stesse 262.508 righe development e stessi tre fold esterni. Due fold interni per selezionare ogni soglia. La media usa la soglia ottimizzata sulle probabilità medie INTERNE, mai la media delle soglie individuali.

Tre fit esterni Phase 14 riutilizzati dopo verifica numerica. Sei fit interni recuperati: soglie e metriche storiche riprodotte esattamente. Nove nuovi fit per il secondo seed. Nessun nuovo training richiesto per calcolare la media.

## Risultati annidati

| Metrica | Phase 14 | Secondo seed | Media uniforme |
| --- | ---: | ---: | ---: |
| f1 | 0.442155227 | 0.439923622 | 0.441111283 |
| precision | 0.367632339 | 0.375140761 | 0.366530566 |
| recall | 0.554572908 | 0.531751510 | 0.553796376 |
| average_precision | 0.428609250 | 0.428760111 | 0.429367500 |
| log_loss | 0.215225071 | 0.215221801 | 0.215085280 |

## Secondo seed

| Fold | F1 | Delta Phase 14 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.435294765 | -0.003732082 | 0.227453233 |
| 2 | 0.442490920 | -0.001562601 | 0.215864448 |
| 3 | 0.441795157 | -0.001716489 | 0.216838285 |

Delta F1 -0.002231605; fold favorevoli 0/3. Criterio del piano non soddisfatto.

Intervallo bootstrap appaiato descrittivo 95%: [-0.004167598; -0.000292364].

| Esito | Phase 14 | Candidato | Differenza |
| --- | ---: | ---: | ---: |
| true_positives | 12855 | 12326 | -529 |
| false_negatives | 10325 | 10854 | +529 |
| false_positives | 22112 | 20531 | -1581 |
| true_negatives | 217216 | 218797 | +1581 |

Record: [20260930T180910156200Z_phase26-phase14-seed20260921-depth5-features128-200trees.json](../experiments/20260930T180910156200Z_phase26-phase14-seed20260921-depth5-features128-200trees.json).

## Media uniforme

| Fold | F1 | Delta Phase 14 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.436783870 | -0.002242977 | 0.197498017 |
| 2 | 0.443951165 | -0.000102356 | 0.217533467 |
| 3 | 0.442919723 | -0.000591922 | 0.216064903 |

Delta F1 -0.001043944; fold favorevoli 0/3. Criterio del piano non soddisfatto.

Intervallo bootstrap appaiato descrittivo 95%: [-0.002308415; +0.000213879].

| Esito | Phase 14 | Candidato | Differenza |
| --- | ---: | ---: | ---: |
| true_positives | 12855 | 12837 | -18 |
| false_negatives | 10325 | 10343 | +18 |
| false_positives | 22112 | 22186 | +74 |
| true_negatives | 217216 | 217142 | -74 |

Record: [20260930T181509840904Z_phase26-phase14-uniform-mean-two-seeds.json](../experiments/20260930T181509840904Z_phase26-phase14-uniform-mean-two-seeds.json).

## Complementarità e limiti

Correlazione delle probabilità dei due seed: 0.995288962; differenza assoluta media: 0.006268225. Decisioni discordanti: 4458, di cui 1037 nei positivi e 3421 nei negativi. Le decisioni usano le rispettive soglie interne: il disaccordo riflette sia score sia soglia.

Bootstrap: 20.000 ripetizioni, seed 20260929, modelli e soglie fissi; non misura la variabilità da riaddestramento o dalla selezione ripetuta dei candidati. La validazione annidata protegge la scelta della soglia; i fold development sono stati usati per più confronti e non costituiscono un test finale intatto.

## Verifiche e costo

15 nuovi fit completati in 273.63 minuti, esclusi controlli preliminari e finali. Probabilità dei checkpoint verificate esattamente; righe, etichette, fold interni, hash e soglie ricontrollati. Componenti e media conservano OOF esterni e interni.

Ambiente: Python 3.14.3, NumPy 2.4.2. Compatibilità con Python 3.9 / NumPy 1.23.1 ancora da verificare.

120 test superati prima dei training; 124 dopo la correzione del controllo sui metadati. I vecchi checkpoint Phase 14 non conservano il campo `output_feature_names`: il loro ordine è verificato mediante preprocessing e bin congelati e probabilità riprodotte esattamente. Per i nuovi checkpoint il campo resta obbligatorio. La correzione riguarda la verifica finale e non ha richiesto nuovi training. Runner eseguito e preflight originali sono archiviati separatamente dal runner di verifica corretto; i relativi hash sono nel confronto JSON.

Il log di training originale conserva l'errore del primo controllo finale. L'output del successivo comando `compare`, concluso con esito 0, è salvato localmente in `results/boosting_artifacts/phase26_two_seeds/verification_stdout.log` (artefatto ignorato da Git). Gli esiti distribuiti sono nel [confronto JSON](../experiments/phase26_two_seeds_comparison.json).

## Decisione e confronto tra i componenti

Conservare **Phase 14, seed 20260920, L2=1**. Entrambi i candidati perdono F1 in tutti e tre i fold e non soddisfano il criterio del piano. La media rispetto a Phase 14 perde 18 veri positivi e aggiunge 74 falsi positivi. Il miglioramento di AP (+0,000758) e log loss (−0,000140) non si traduce in un miglioramento dell'F1 annidato.

La media supera il secondo seed singolo di +0,001187661 F1; il vantaggio è positivo in tutti e tre i fold. Il suo intervallo bootstrap descrittivo 95% rispetto al secondo seed è [−0,000488604; +0,002864304] e comprende zero. Il primo seed rimane migliore di entrambi. Non ampliare ora una griglia di seed sulla base di questa coppia.

## Nota sui tempi

Il tempo registrato dei training è **273,63 minuti, circa 4 ore e 34 minuti**. È dominato dal primo fit esterno del nuovo seed, che ha registrato 215,28 minuti: il log passa da 175 alberi a 529,5 secondi a 200 alberi a 12.263,3 secondi. In seguito il tempo CPU osservato per l'intero processo era circa 35 minuti, molto inferiore al tempo trascorso. La causa dell'intervallo anomalo non è stata stabilita. Gli altri fit e il preprocessing hanno richiesto complessivamente 58,35 minuti. Questo tempo totale non rappresenta il costo abituale della media di due seed. Preparazione e verifiche finali sono escluse dal valore dei training.

Il confronto successivo con learning rate 0,025 e 400 alberi contro Phase 14 è stato completato nella [Phase 27](phase27_phase14_lr0025_400trees_summary.md) e non è stato adottato. Configurazione finale e soglia di submission restano invariate.

Confronto completo: [JSON](../experiments/phase26_two_seeds_comparison.json). Controlli: [preflight](phase26_two_seeds_preflight.json).
Configurazione del secondo seed: [JSON](../../configs/experiments/phase26_phase14_seed20260921.json).
