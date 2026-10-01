# Phase 29 — Selezione di 100 e 60 feature

## Risultati

Controllo Phase 14: F1 annidato **0.442155227**.

| Input | F1 | Delta F1 | Precision | Recall | AP | Log loss | Fold favorevoli |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.439223556 | -0.002931671 | 0.366645 | 0.547627 | 0.427210 | 0.215535 | 0/3 |
| 60 | 0.437584073 | -0.004571154 | 0.358536 | 0.561346 | 0.423974 | 0.216362 | 0/3 |

## Protocollo e interpretazione

Ogni training esterno e interno costruisce la propria selezione: booster Phase 14 a 200 alberi allenato sull'80% del relativo training, con holdout stratificato del 20%. Il preprocessing del selettore viene appreso solo sulla sua parte di fit. La soglia viene scelta una volta sul suo holdout originale; non cambia durante le permutazioni. L'importanza usa fino a 10.000 righe del medesimo holdout, campionate mantenendo la proporzione delle classi, e tre permutazioni, compresi i NaN.

Ordine fissato prima dei risultati: calo medio F1, poi calo AP, aumento log loss, numero di split e ordine originale, usati come spareggi esatti. La deviazione standard descrive l'instabilità: tre ripetizioni non sono una prova statistica. Le permutazioni dei gruppi semantici sono diagnostiche e non cambiano il ranking o il budget esatto delle colonne. Non si eliminano colonne in base alla sola percentuale di mancanti.

Top 60 è un sottoinsieme di top 100 in ogni training. Il booster ridotto viene riaddestrato su tutto quel training. Max_features resta la frazione 128/295: 44 colonne candidate con 100 input, 27 con 60. Soglie finali scelte da OOF dei due fold interni; tre fold esterni e tutti i seed di controllo invariati.

I valori di importanza descrivono il modello del selettore. Possono sottovalutare variabili ridondanti o utili dopo un riaddestramento; il confronto annidato verifica la pipeline di selezione, non dimostra che sia stata trovata la migliore combinazione possibile.

## Dettaglio top 100

Record: [20260930T222023662096Z_phase29-phase14-fold-local-top100-permutation-features.json](../../results/experiments/phase29_feature_selection/20260930T222023662096Z_phase29-phase14-fold-local-top100-permutation-features.json).

Regola di promozione del piano soddisfatta: **no**. Il modello finale non viene modificato automaticamente.

Delta F1 per fold: -0.004166111, -0.002138454, -0.002742295.
Delta dei conteggi: {'true_positives': -161, 'false_positives': -184, 'true_negatives': 184, 'false_negatives': 161}.
Intervallo bootstrap descrittivo 95% del delta F1: [-0.004927242814316479, -0.0009424799522040488]. Fit e soglie fissi; non comprende l'incertezza del training e delle molte prove sul development.
Feature comuni alle tre selezioni esterne: 37/100. Jaccard fra coppie: [0.3793103448275862, 0.3986013986013986, 0.3986013986013986].
Tempo dei nove fit ridotti e preprocessing: 22.25 minuti; costo dei nove selettori condiviso: 46.89 minuti.

## Dettaglio top 60

Record: [20260930T222046496842Z_phase29-phase14-fold-local-top60-permutation-features.json](../../results/experiments/phase29_feature_selection/20260930T222046496842Z_phase29-phase14-fold-local-top60-permutation-features.json).

Regola di promozione del piano soddisfatta: **no**. Il modello finale non viene modificato automaticamente.

Delta F1 per fold: -0.001707034, -0.009785136, -0.002303613.
Delta dei conteggi: {'true_positives': 157, 'false_positives': 1168, 'true_negatives': -1168, 'false_negatives': -157}.
Intervallo bootstrap descrittivo 95% del delta F1: [-0.006783863388137183, -0.0023207463111974587]. Fit e soglie fissi; non comprende l'incertezza del training e delle molte prove sul development.
Feature comuni alle tre selezioni esterne: 19/60. Jaccard fra coppie: [0.34831460674157305, 0.30434782608695654, 0.31868131868131866].
Tempo dei nove fit ridotti e preprocessing: 13.51 minuti; costo dei nove selettori condiviso: 46.89 minuti.

## Classifica e liste sul development

Questa decima selezione usa soltanto le 262.508 righe development. Produce liste utilizzabili come punto di partenza per un training successivo; non vengono riutilizzate per calcolare gli F1 riportati sopra. Un futuro training sull'intero dataset deve rifare la selezione su quel training.

- [Classifica completa](phase29_feature_selection/development/ranking.csv).
- [Lista delle 100 feature](phase29_feature_selection/development/top100.json).
- [Lista delle 60 feature](phase29_feature_selection/development/top60.json).
- [Importanza dei gruppi](phase29_feature_selection/development/group_importance.json).

| Posizione | Feature | Calo medio F1 | Deviazione standard | Calo AP | Mancanti nel fit |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | SEX | 0.037302 | 0.007413 | 0.036342 | 0.00% |
| 2 | GENHLTH | 0.020772 | 0.004291 | 0.022476 | 0.26% |
| 3 | _AGE80 | 0.018199 | 0.006575 | 0.014648 | 0.00% |
| 4 | HAREHAB1 | 0.014587 | 0.000564 | 0.068743 | 99.77% |
| 5 | FC60_ | 0.012259 | 0.002375 | 0.000779 | 0.00% |
| 6 | MAXVO2_ | 0.009136 | 0.003564 | 0.001709 | 0.00% |
| 7 | CVDSTRK3 | 0.008628 | 0.001859 | 0.014033 | 0.22% |
| 8 | TOLDHI2 | 0.006945 | 0.000772 | 0.001860 | 14.10% |
| 9 | _AGE_G | 0.006451 | 0.003032 | 0.001907 | 0.00% |
| 10 | _RFCHOL | 0.005971 | 0.002054 | 0.006192 | 14.10% |
| 11 | BPHIGH4 | 0.005410 | 0.001544 | 0.000457 | 0.29% |
| 12 | CVDASPRN | 0.005299 | 0.002462 | 0.005048 | 95.68% |
| 13 | CHOLCHK | 0.005237 | 0.006324 | 0.003073 | 14.63% |
| 14 | _RFHYPE5 | 0.005126 | 0.001419 | 0.001147 | 0.29% |
| 15 | _RFHLTH | 0.004908 | 0.006011 | 0.012113 | 0.26% |
| 16 | DIFFWALK | 0.004741 | 0.001473 | 0.000698 | 3.31% |
| 17 | QLACTLM2 | 0.004534 | 0.002058 | 0.001206 | 2.79% |
| 18 | CHCCOPD1 | 0.004527 | 0.003541 | 0.009234 | 0.47% |
| 19 | BPMEDS | 0.004361 | 0.002044 | 0.011640 | 59.91% |
| 20 | SMOKE100 | 0.003464 | 0.002484 | 0.000113 | 3.99% |
| 21 | INCOME2 | 0.003331 | 0.001508 | 0.000199 | 17.86% |
| 22 | IDAY | 0.002546 | 0.000661 | 0.000174 | 0.00% |
| 23 | DRNK3GE5 | 0.002543 | 0.001619 | -0.000123 | 52.83% |
| 24 | EMPLOY1 | 0.002538 | 0.002985 | -0.000709 | 0.85% |
| 25 | PNEUVAC3 | 0.002380 | 0.004288 | 0.004762 | 17.62% |
| 26 | PERSDOC2 | 0.002158 | 0.003163 | 0.003372 | 0.41% |
| 27 | PHYSHLTH | 0.002092 | 0.002181 | 0.001189 | 2.15% |
| 28 | _SMOKER3 | 0.002075 | 0.003497 | -0.000071 | 4.09% |
| 29 | FLUSHOT6 | 0.002028 | 0.000628 | 0.000175 | 10.07% |
| 30 | ARTHDIS2 | 0.002024 | 0.000120 | -0.000052 | 70.04% |

La selezione non risolve il dubbio di ammissibilità di HAREHAB1. Il 20% storico esterno al development non è stato consultato in questa prova. I fold development sono già stati usati per molti esperimenti: i risultati non equivalgono a una valutazione finale indipendente.

## Artefatti e isolamento

Tempo totale di questa esecuzione: 93.96 minuti. Dieci selettori e diciotto fit dei candidati completati, in sequenza, con un thread NumPy e priorità inferiore su Windows.

Artefatti numerici e tutti i checkpoint: `results/boosting_artifacts/phase29_phase14_feature_selection`. Ogni selettore conserva le righe di fit, holdout, permutazione ed esclusione; ogni candidato conserva lista, righe, etichette, probabilità e checkpoint. Tutti i diciotto checkpoint riproducono esattamente le probabilità salvate. OOF esterni e interni, sorgente eseguito, preflight e hash conservati.

I sorgenti, le configurazioni e gli input protetti di Phase 27/28 sono stati verificati prima e dopo; nessuna scrittura nelle loro cartelle o nei registri condivisi. I due nuovi record restano nel registro dedicato `results/experiments/phase29_feature_selection/`. Nessuna nuova submission, modifica del modello finale, commit o push.

## Stabilità della selezione e variabili rare

[Frequenza di selezione delle 295 colonne](phase29_feature_selection/selection_frequency.csv). È una diagnosi delle liste dei diversi training: non è una nuova lista da applicare globalmente agli stessi fold. Solo 37 colonne su 100 e 19 su 60 sono comuni a tutti e tre i training esterni.

Sul development, il taglio delle 100 colonne cade su CPDEMO1 (calo F1 0.000347, deviazione standard 0.000120); quello delle 60 cade su PDIABTST (calo F1 0.000828, deviazione standard 0.000711). Le importanze vicino al taglio sono piccole e vanno lette insieme alla loro variabilità.

HAREHAB1 è quarta nella classifica development ed è inclusa in entrambe le liste esterne di tutti i fold. I sei selettori interni non la utilizzano: hanno meno di 200 risposte note, sotto il minimo foglia. Nei training interni completi il supporto può essere diverso; questa diagnosi spiega la differenza di ranking e non identifica da sola la causa della perdita F1.

| Partizione | Risposte note nel training completo | Risposte note nel fit del selettore | Split HAREHAB1 | Posizione |
| --- | ---: | ---: | ---: | ---: |
| development | 607 | 488 | 106 | 4 |
| outer_1 | 384 | 302 | 90 | 3 |
| outer_1_inner_1 | 188 | 153 | 0 | 170 |
| outer_1_inner_2 | 196 | 163 | 0 | 165 |
| outer_2 | 405 | 331 | 94 | 3 |
| outer_2_inner_1 | 196 | 154 | 0 | 197 |
| outer_2_inner_2 | 209 | 165 | 0 | 193 |
| outer_3 | 425 | 334 | 92 | 3 |
| outer_3_inner_1 | 206 | 163 | 0 | 160 |
| outer_3_inner_2 | 219 | 167 | 0 | 191 |

## Importanza diagnostica dei gruppi sul development

Le colonne di ciascun gruppo vengono mescolate insieme. Il calo F1 dipende dal numero di variabili e dalle loro relazioni: non è un effetto causale e non equivale al vantaggio di rimuovere il gruppo. I gruppi non determinano le liste top100/top60.

| Gruppo | Calo medio F1 | Deviazione standard |
| --- | ---: | ---: |
| age | 0.021107 | 0.007160 |
| hypertension | 0.007624 | 0.001154 |
| diabetes | 0.005926 | 0.001235 |
| smoking | 0.005869 | 0.001588 |
| body_size | 0.001511 | 0.000567 |
| exercise_frequency_1 | 0.001226 | 0.001500 |
| exercise_summary | 0.000385 | 0.000803 |
| exercise_frequency_2 | 0.000310 | 0.000371 |
| alcohol | 0.000054 | 0.003137 |
| diet | -0.000312 | 0.003662 |

## Decisione

Le due pipeline di selezione testate sono concluse e non adottate: perdono F1 su tutti e tre i fold. Gli intervalli bootstrap condizionali sono interamente negativi, ma non comprendono il nuovo training o la scelta ripetuta degli esperimenti. Non è dimostrato che ogni possibile selezione di 100 o 60 colonne peggiori. Non è stata avviata la variante con 30 colonne, che era subordinata a un segnale favorevole delle prime riduzioni.

## Verifica finale

137 test superati prima del training e 138 dopo. Le venti liste pubbliche top100/top60 coincidono con le selezioni private effettivamente usate. Tutti i record dei due registri hanno ID univoci e hash corretti; i collegamenti del rapporto esistono. Sorgenti, dati, configurazione finale e input protetti Phase 27/28 sono invariati. Il piano e lo storico sono stati aggiornati soltanto dopo la conclusione delle tre esecuzioni.
