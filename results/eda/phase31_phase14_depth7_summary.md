# Phase 31 — Profondità 7

Profondità massima da 5 a 7 con peso dei positivi 1, 200 alberi, learning rate 0,05, 295 feature e altri parametri Phase 14 invariati.

Controllo: Phase 14. Stesse righe development, tre fold esterni e due interni per scegliere ogni soglia. Seed invariati. Nove nuovi fit del candidato; baseline verificata riutilizzata.

| Metrica | Phase 14 | Candidato | Delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.876434242 | 0.874940954 | -0.001493288 |
| precision | 0.367632339 | 0.364346970 | -0.003285369 |
| recall | 0.554572908 | 0.559016393 | +0.004443486 |
| f1 | 0.442155227 | 0.441160950 | -0.000994277 |
| specificity | 0.907607969 | 0.905539678 | -0.002068291 |
| balanced_accuracy | 0.731090438 | 0.732278036 | +0.001187597 |
| average_precision | 0.428609250 | 0.428748338 | +0.000139087 |
| log_loss | 0.215225071 | 0.215090233 | -0.000134838 |

| Fold | F1 candidato | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.436892735 | -0.002134113 | 0.197111771 |
| 2 | 0.444066355 | +0.000012834 | 0.220880536 |
| 3 | 0.442909091 | -0.000602555 | 0.215307776 |

| Esito | Phase 14 | Candidato |
| --- | ---: | ---: |
| false_negatives | 10325 | 10222 |
| false_positives | 22112 | 22607 |
| true_negatives | 217216 | 216721 |
| true_positives | 12855 | 12958 |

Criterio operativo del piano: non soddisfatto; fold favorevoli 1/3. Configurazione finale invariata.

Bootstrap descrittivo del delta F1: [-0.002978237; +0.001004398], 20.000 ripetizioni con modelli e soglie fissi. Non misura l'incertezza da riaddestramento o selezione ripetuta.

Tempo della CV: 63.89 minuti, con un altro esperimento eseguito in parallelo. Preflight e verifiche finali esclusi. Ambiente: {'numpy': '2.4.2', 'python': '3.14.3'}.

Nove partizioni verificate; colonne e bin invariati salvo la modifica richiesta. Checkpoint, probabilità, righe, hash, soglie interne e metriche riprodotti. Il 20% iniziale non è utilizzato; i fold development sono già stati consultati per più esperimenti.

Record: results/boosting_artifacts/phase31_phase14_depth7/records/20261001T001533065665Z_phase31-phase14-depth7-depth7-features128-200trees.json
Artefatti e confronto: results/boosting_artifacts/phase31_phase14_depth7/
