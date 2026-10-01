# Phase 32 — MLP con preprocessing dedicato

Valutazione completa: 262,508 righe development, tre fold esterni e due interni. Soglie F1 scelte dagli OOF interni. Epoch selection su holdout 10% del training, quindi refit completo.

| Modello | F1 | Delta Phase 14 | Precision | Recall | AP | Log loss | Fold favorevoli |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 14 | 0.442155227 | — | 0.367632 | 0.554573 | 0.428609 | 0.215225 | — |
| mlp64_32 | 0.424451233 | -0.017703995 | 0.360722 | 0.515531 | 0.387782 | 0.221674 | 0/3 |
| mlp128_64 | 0.419267861 | -0.022887366 | 0.340005 | 0.546721 | 0.379380 | 0.222710 | 0/3 |

| Modello / fold | F1 | Delta Phase 14 | Soglia interna | Epoche del refit | Input |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp64_32 / 1 | 0.411656403 | -0.027370445 | 0.216357470 | 7 | 1178 |
| mlp64_32 / 2 | 0.429213127 | -0.014840394 | 0.217222542 | 7 | 1180 |
| mlp64_32 / 3 | 0.429601553 | -0.013910093 | 0.202820331 | 5 | 1179 |
| mlp128_64 / 1 | 0.417129481 | -0.021897366 | 0.208536416 | 2 | 1178 |
| mlp128_64 / 2 | 0.422633317 | -0.021420204 | 0.190321296 | 4 | 1180 |
| mlp128_64 / 3 | 0.417587727 | -0.025923919 | 0.227644756 | 5 | 1179 |

Durata della suite condivisa: 8.86 minuti. Include preprocessing, 18 selezioni delle epoche e 18 refit; non è il tempo individuale di ciascun candidato.

Dati, etichette, metadati e split coincidono con la baseline; F1 storico ricalcolato dagli OOF. Tutti i 18 checkpoint di refit riproducono gli score e le classificazioni salvate. Confini gradient fit/early stopping/evaluation, soglie, metriche e hash verificati. I sette file protetti del booster conservano gli hash del preflight.

La MLP usa 295 sorgenti più il flag ausiliario di età del piano. La pulizia del codebook è condivisa; imputazione, scaling, one-hot, missing/unseen flags e frequenze convertite sono specifici della MLP. Il confronto misura le pipeline complete, non il solo effetto della famiglia del modello.

Entrambe le MLP hanno F1 inferiore a Phase 14 nei tre fold. HAREHAB1 è presente; il suo legame con il target è descritto nel [report finale](../../docs/FINAL_REPORT.md#harehab1). Ambiente eseguito: Python 3.14.3 / NumPy 2.4.2, OpenBLAS con un thread; compatibilità nell'ambiente ufficiale non verificata.

Phase 14 rimane il modello selezionato. Dettagli: [metodo MLP](../../docs/MLP.md), [confronto completo](../experiments/mlp/phase32_comparison.json). I checkpoint locali sono in `results/mlp_artifacts/phase32/`.
