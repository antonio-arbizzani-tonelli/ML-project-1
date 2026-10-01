# Phase 28 — Rimozione della sola ALCDAY5

Rimuovere solo ALCDAY5, mantenendo DROCDY3_ e tutte le altre feature Phase 14.

Controllo: Phase 14. Stesse righe development, tre fold esterni e due interni per scegliere ogni soglia. Seed invariati. Nove nuovi fit del candidato; baseline verificata riutilizzata.

| Metrica | Phase 14 | Candidato | Delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.876434242 | 0.873036250 | -0.003397992 |
| precision | 0.367632339 | 0.360563845 | -0.007068494 |
| recall | 0.554572908 | 0.566091458 | +0.011518550 |
| f1 | 0.442155227 | 0.440535142 | -0.001620086 |
| specificity | 0.907607969 | 0.902765243 | -0.004842726 |
| balanced_accuracy | 0.731090438 | 0.734428350 | +0.003337912 |
| average_precision | 0.428609250 | 0.429039626 | +0.000430376 |
| log_loss | 0.215225071 | 0.215250685 | +0.000025614 |

| Fold | F1 candidato | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.435323752 | -0.003703096 | 0.198075855 |
| 2 | 0.441648634 | -0.002404887 | 0.205198772 |
| 3 | 0.444821383 | +0.001309737 | 0.209596383 |

| Esito | Phase 14 | Candidato |
| --- | ---: | ---: |
| false_negatives | 10325 | 10058 |
| false_positives | 22112 | 23271 |
| true_negatives | 217216 | 216057 |
| true_positives | 12855 | 13122 |

Criterio operativo del piano: non soddisfatto; fold favorevoli 1/3. Configurazione finale invariata.

Bootstrap descrittivo del delta F1: [-0.003295025; +0.000067219], 20.000 ripetizioni con modelli e soglie fissi. Non misura l'incertezza da riaddestramento o selezione ripetuta.

Tempo della CV: 57.84 minuti, con un altro esperimento eseguito in parallelo. Preflight e verifiche finali esclusi. Ambiente: {'numpy': '2.4.2', 'python': '3.14.3'}.

Nove partizioni verificate; colonne e bin invariati salvo la modifica richiesta. Checkpoint, probabilità, righe, hash, soglie interne e metriche riprodotti. Il 20% iniziale non è utilizzato; i fold development sono già stati consultati per più esperimenti.

Record: results/boosting_artifacts/phase28_phase14_ablate_alcday5/records/20260930T212010451103Z_phase28-phase14-ablate-alcday5-depth5-features128-200trees.json
Artefatti e confronto: results/boosting_artifacts/phase28_phase14_ablate_alcday5/
