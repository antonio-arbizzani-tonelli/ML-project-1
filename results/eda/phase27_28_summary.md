# Phase 27–28 — Confronto finale con Phase 14

Entrambe le prove sono concluse e verificate. Conservare Phase 14: nessuno dei candidati migliora l’F1 aggregato e almeno due fold su tre.

Phase 27 modifica solo learning rate (0,05 → 0,025) e alberi (200 → 400), con 295 feature. Phase 28 rimuove solo ALCDAY5, mantiene DROCDY3_ e tutti i parametri originali, con 294 feature. Non è stata provata la combinazione.

262.508 righe development, stessi split e seed; tre fold esterni e due interni per scegliere ogni soglia. Diciotto nuovi fit in totale. Baseline verificata riutilizzata.

| Metrica | Phase 14 | Phase 27 | Phase 28 |
| --- | ---: | ---: | ---: |
| f1 | 0.442155227 | 0.440587222 | 0.440535142 |
| precision | 0.367632339 | 0.364261750 | 0.360563845 |
| recall | 0.554572908 | 0.557377049 | 0.566091458 |
| average_precision | 0.428609250 | 0.429352055 | 0.429039626 |
| log_loss | 0.215225071 | 0.215123329 | 0.215250685 |

| Fold | F1 Phase 14 | F1 Phase 27 | Delta 27 | F1 Phase 28 | Delta 28 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.439026847 | 0.435906570 | -0.003120278 | 0.435323752 | -0.003703096 |
| 2 | 0.444053521 | 0.443457606 | -0.000595915 | 0.441648634 | -0.002404887 |
| 3 | 0.443511646 | 0.442440913 | -0.001070733 | 0.444821383 | +0.001309737 |

| Fold | Soglia Phase 14 | Soglia Phase 27 | Soglia Phase 28 |
| --- | ---: | ---: | ---: |
| 1 | 0.203056689 | 0.206211866 | 0.198075855 |
| 2 | 0.220104637 | 0.210838301 | 0.205198772 |
| 3 | 0.209445973 | 0.208149784 | 0.209596383 |

Le soglie sono selezionate esclusivamente sugli OOF interni. Ogni F1 esterno usa la soglia del rispettivo fold; i valori non sono nuove soglie di submission.

Phase 27: F1 0,440587222 (delta −0,001568005), nessun fold favorevole. Aumenta i veri positivi di 65, ma aggiunge 437 falsi positivi. Il lieve aumento di recall e AP e la lieve diminuzione di log loss non si traducono in un aumento dell’F1. Il costo di 400 alberi non è giustificato da questo confronto.

Phase 28: F1 0,440535142 (delta −0,001620086), un fold favorevole su tre. Aumenta i veri positivi di 267, ma aggiunge 1.159 falsi positivi. Il recall sale, mentre la precision scende abbastanza da ridurre l’F1. L’ablazione non offre un vantaggio coerente e ALCDAY5 resta nel modello. Il confronto misura l’effetto complessivo della pipeline, inclusi il campionamento delle feature e le nuove soglie; non prova un’utilità causale della variabile.

Bootstrap appaiato descrittivo, Phase 27: [-0.003124704; -0.000038643].
Bootstrap appaiato descrittivo, Phase 28: [-0.003295025; +0.000067219].

20.000 ripetizioni, seed 20260929, modelli e soglie fissi. Phase 27 ha un intervallo interamente negativo, con limite superiore vicino a zero; quello di Phase 28 comprende zero. Non misurano l’incertezza da riaddestramento né da selezione ripetuta sui fold development. Questi risultati giustificano mantenere il controllo, non affermare che tutte le varianti vicine debbano peggiorare.

CV Phase 27: 111.01 minuti. CV Phase 28: 57.84 minuti. La durata trascorsa del coordinatore è 111.50 minuti, inclusa la verifica finale. Avvio alle 22:22 del 30 settembre, conclusione di Phase 28 alle 23:20 del 30 settembre e di Phase 27 alle 00:13 del 1 ottobre 2026, fuso Europe/Rome. Preflight esclusi dai tempi CV. Le prove hanno condiviso risorse durante la prima parte: non è un confronto misurato con esecuzione sequenziale.

124 test esistenti superati prima del training; otto cambiamenti accidentali respinti dai controlli delle configurazioni. Preflight su tutte le nove partizioni per candidato. Probabilità dei tre checkpoint identiche agli OOF per ciascuna prova, righe e hash coerenti, soglie e metriche ricalcolate dagli OOF interni. Entrambi i processi terminati con codice 0; record pubblicati in sequenza nel registro condiviso. Nessun riaddestramento richiesto.

Decisione: conservare Phase 14, learning rate 0,05, 200 alberi, L2=1, seed 20260920 e ALCDAY5. Nessuna nuova submission e nessuna combinazione adottata. La compatibilità con l’ambiente ufficiale resta da verificare prima della consegna.

- [Record Phase 27](../experiments/20260930T221319968519Z_phase27-phase14-lr0025-400trees-depth5-features128-400trees.json).
- [Confronto Phase 27](../experiments/phase27_comparison.json).
- [Record Phase 28](../experiments/20260930T212010451103Z_phase28-phase14-ablate-alcday5-depth5-features128-200trees.json).
- [Confronto Phase 28](../experiments/phase28_comparison.json).
