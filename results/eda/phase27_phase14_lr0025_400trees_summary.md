# Phase 27 — Learning rate 0,025 e 400 alberi

Confrontare learning rate 0,025 e 400 alberi con Phase 14, mantenendo invariati preprocessing, altri parametri, split e seed.

Controllo: Phase 14. Stesse righe development, tre fold esterni e due interni per scegliere ogni soglia. Seed invariati. Nove nuovi fit del candidato; baseline verificata riutilizzata.

| Metrica | Phase 14 | Candidato | Delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.876434242 | 0.875017142 | -0.001417100 |
| precision | 0.367632339 | 0.364261750 | -0.003370589 |
| recall | 0.554572908 | 0.557377049 | +0.002804142 |
| f1 | 0.442155227 | 0.440587222 | -0.001568005 |
| specificity | 0.907607969 | 0.905782023 | -0.001825946 |
| balanced_accuracy | 0.731090438 | 0.731579536 | +0.000489098 |
| average_precision | 0.428609250 | 0.429352055 | +0.000742805 |
| log_loss | 0.215225071 | 0.215123329 | -0.000101742 |

| Fold | F1 candidato | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.435906570 | -0.003120278 | 0.206211866 |
| 2 | 0.443457606 | -0.000595915 | 0.210838301 |
| 3 | 0.442440913 | -0.001070733 | 0.208149784 |

| Esito | Phase 14 | Candidato |
| --- | ---: | ---: |
| false_negatives | 10325 | 10260 |
| false_positives | 22112 | 22549 |
| true_negatives | 217216 | 216779 |
| true_positives | 12855 | 12920 |

Criterio operativo del piano: non soddisfatto; fold favorevoli 0/3. Configurazione finale invariata.

Bootstrap descrittivo del delta F1: [-0.003124704; -0.000038643], 20.000 ripetizioni con modelli e soglie fissi. Non misura l'incertezza da riaddestramento o selezione ripetuta.

Tempo della CV: 111.01 minuti, con un altro esperimento eseguito in parallelo. Preflight e verifiche finali esclusi. Ambiente: {'numpy': '2.4.2', 'python': '3.14.3'}.

Nove partizioni verificate; colonne e bin invariati salvo la modifica richiesta. Checkpoint, probabilità, righe, hash, soglie interne e metriche riprodotti. Il 20% iniziale non è utilizzato; i fold development sono già stati consultati per più esperimenti.

Record: results/boosting_artifacts/phase27_phase14_lr0025_400trees/records/20260930T221319968519Z_phase27-phase14-lr0025-400trees-depth5-features128-400trees.json
Artefatti e confronto: results/boosting_artifacts/phase27_phase14_lr0025_400trees/
