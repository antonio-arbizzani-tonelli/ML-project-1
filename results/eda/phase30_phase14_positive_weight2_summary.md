# Phase 30 — Peso dei positivi 2

Peso 2 per i positivi e 1 per i negativi nella loss logistica, gradienti, Hessiani e prevalenza iniziale; metriche e selezione F1 non pesate. Tutti gli altri parametri e le 295 feature restano Phase 14.

Controllo: Phase 14. Stesse righe development, tre fold esterni e due interni per scegliere ogni soglia. Seed invariati. Nove nuovi fit del candidato; baseline verificata riutilizzata.

| Metrica | Phase 14 | Candidato | Delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.876434242 | 0.873676231 | -0.002758011 |
| precision | 0.367632339 | 0.361155162 | -0.006477177 |
| recall | 0.554572908 | 0.560008628 | +0.005435720 |
| f1 | 0.442155227 | 0.439118448 | -0.003036779 |
| specificity | 0.907607969 | 0.904056358 | -0.003551611 |
| balanced_accuracy | 0.731090438 | 0.732032493 | +0.000942055 |
| average_precision | 0.428609250 | 0.428237932 | -0.000371318 |
| log_loss | 0.215225071 | 0.231722615 | +0.016497544 |

| Fold | F1 candidato | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.433710992 | -0.005315855 | 0.331399741 |
| 2 | 0.440990968 | -0.003062553 | 0.338300890 |
| 3 | 0.442857143 | -0.000654503 | 0.351041141 |

| Esito | Phase 14 | Candidato |
| --- | ---: | ---: |
| false_negatives | 10325 | 10199 |
| false_positives | 22112 | 22962 |
| true_negatives | 217216 | 216366 |
| true_positives | 12855 | 12981 |

Criterio operativo del piano: non soddisfatto; fold favorevoli 0/3. Configurazione finale invariata.

Bootstrap descrittivo del delta F1: [-0.004730150; -0.001318961], 20.000 ripetizioni con modelli e soglie fissi. Non misura l'incertezza da riaddestramento o selezione ripetuta.

Tempo della CV: 48.55 minuti, con un altro esperimento eseguito in parallelo. Preflight e verifiche finali esclusi. Ambiente: {'numpy': '2.4.2', 'python': '3.14.3'}.

Nove partizioni verificate; colonne e bin invariati salvo la modifica richiesta. Checkpoint, probabilità, righe, hash, soglie interne e metriche riprodotti. Il 20% iniziale non è utilizzato; i fold development sono già stati consultati per più esperimenti.

Record: results/boosting_artifacts/phase30_phase14_positive_weight2/records/20261001T000013003849Z_phase30-phase14-positive-weight2-depth5-features128-200trees.json
Artefatti e confronto: results/boosting_artifacts/phase30_phase14_positive_weight2/
