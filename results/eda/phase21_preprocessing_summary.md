# Phase 21 — prova mirata

## Protocollo

Due candidati valutati separatamente rispetto a Phase 14. La Phase 21 conserva esattamente le quattro risposte di BPHIGH4 e DIABETE3 e usa i quantili originali sulle altre 293 colonne. La Phase 22 parte direttamente da Phase 14 e sostituisce EXRACT11 e EXRACT21 con indicatori per tutte le categorie osservate nel training; il binning esatto riguarda soltanto questi nuovi indicatori.

Vocabolario appreso dentro ogni training, senza usare la validazione. I mancanti restano NaN in tutto il blocco di attività; una colonna unknown distingue eventuali risposte finite mai osservate nel training. EXRACT21=88 resta una categoria valida. Non si raggruppano categorie rare in questa prima prova.

Ogni candidato usa i 262.508 esempi development, tre fold esterni e due interni per selezionare la soglia F1: nove fit da 200 alberi. Stessi seed, profondità 5, learning rate 0,05, minimo foglia 200, L2=1 e 128 feature candidate per nodo. Il 20% iniziale non è usato in queste prove; era già stato consultato storicamente. HAREHAB1 resta inclusa come nel controllo.

Prima dei training passano 101 test, inclusi vocabolario dal solo training, conservazione dei mancanti, selezione delle colonne e continuazione dei checkpoint. Il preflight verifica tutti i nove training di ciascun candidato, categorie, bin, mancanti e valori delle altre feature. Le predizioni dei tre checkpoint originali Phase 14 coincidono esattamente con gli OOF salvati. I profili dei nove fit reali corrispondono al preflight; gli OOF e le predizioni dei checkpoint candidati sono verificati dopo il training.

## Risultati annidati

| Modello | F1 | Delta F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 14 | 0.442155 | — | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Phase 21 | 0.439731 | -0.002424 | 0.361644 | 0.560828 | 0.428817 | 0.215263 |

## Phase 21: dettagli

| Fold | F1 controllo | F1 candidato | Delta | Soglia candidato |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.439027 | 0.435271 | -0.003756 | 0.194159 |
| 2 | 0.444054 | 0.441299 | -0.002754 | 0.210856 |
| 3 | 0.443512 | 0.442973 | -0.000538 | 0.215149 |

| Modello | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 | 12855 | 10325 | 22112 | 217216 |
| Phase 21 | 13000 | 10180 | 22947 | 216381 |

Recupera 436 falsi negativi e perde 291 veri positivi; introduce 1744 falsi positivi e ne elimina 909.

Fold favorevoli: 0/3. Intervallo bootstrap condizionato 95% per delta F1: [-0.004001, -0.000848], a modelli e soglie fissi (20.000 ricampionamenti appaiati stratificati; seed 20260929). Non comprende variabilità del training o selezione ripetuta dei candidati.

Runtime completo: 42.1 minuti. Python 3.14.3, NumPy 2.4.2. Compatibilità nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1 non eseguita.

Non soddisfa la regola operativa del piano: la modifica non viene adottata sul riferimento.

Record completo: [20260930T020213451046Z_phase21-phase14-clinical-exact-depth5-features128-200trees.json](../experiments/20260930T020213451046Z_phase21-phase14-clinical-exact-depth5-features128-200trees.json). Confronto verificato: [JSON](../experiments/phase21_preprocessing_comparison.json).

Variazione netta rispetto a Phase 14: +145 TP e +835 FP. Il guadagno di recall non compensa la perdita di precision.

## Decisione e seguito

La configurazione finale e la soglia di submission Phase 14 sono rimaste invariate durante questi esperimenti, che non hanno addestrato un modello sull'intero training né prodotto una submission.

Entrambe le varianti peggiorano tutti e tre i fold rispetto a Phase 14 e non vengono adottate. Chiudere queste configurazioni precise; le prove non dimostrano che ogni binning esatto o ogni trattamento nominale sia inferiore. Il guadagno di AP è molto piccolo in entrambi i candidati e non compensa il calo della metrica prioritaria F1.

I confronti successivi hanno valutato separatamente i soli indicatori di mancata risposta nella [Phase 23](phase23_missing_reason_flags_summary.md) e l'imputazione nella [Phase 24](phase24_original_restricted_imputation_summary.md); nessuna delle due varianti è stata adottata. Split categorici nativi e campionamento per feature sorgente sono rimasti idee non eseguite. La scelta finale è descritta nel [report del modello](../../docs/FINAL_REPORT.md).

Artefatti locali: [preflight](phase21_22_preflight.json), configurazioni in configs/experiments, OOF e checkpoint in results/boosting_artifacts. Record, confronti e documentazione sono conservati nel progetto.
