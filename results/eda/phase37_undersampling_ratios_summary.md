# Phase 37 — Undersampling casuale 5:1 e 7:1 su Phase 14

Confronto sul miglior modello storico Phase 14 (F1 nested 0,442155227), usando lo snapshot numerico storico già verificato per Phase 35. Tutti i positivi restano nel training. I negativi seguono una permutazione casuale condivisa per ciascun fit: il campione 5:1 è contenuto nel 7:1. Tutte le righe di validazione sono conservate; preprocessing e bin apprendono solo dalle righe selezionate. Soglie scelte tramite OOF interno. Holdout e test non valutati.

Stessi 295 predittori, parametri, seed e partizioni del controllo. Nove fit per candidato; tre esterni e sei interni. Il risultato 3:1 di Phase 36 usa invece la versione corretta Phase 34 e non costituisce un controllo storico comparabile.

| Pipeline storica | F1 | Precision | Recall | AP | Falsi positivi | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tutte le righe — Phase 14 | 0.442155227 | 0.367632339 | 0.554572908 | 0.428609250 | 22112 | +0.000000000 |
| Casuale 5:1 | 0.438225776 | 0.365893131 | 0.546203624 | 0.426827450 | 21942 | -0.003929452 |
| Casuale 7:1 | 0.439819715 | 0.363931740 | 0.555694564 | 0.427890611 | 22513 | -0.002335512 |

## Rapporto 5:1

| Fold | F1 | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.435351040 | -0.003675808 | 0.353960287 |
| 2 | 0.438983318 | -0.005070203 | 0.366027358 |
| 3 | 0.440367921 | -0.003143725 | 0.357943822 |

Fold favorevoli: 0/3. Intervallo condizionale del delta F1: [-0.005761870, -0.002082791].
Training esterni conservati: 92718, 92718, 92724; tutti i positivi. Runner completo: 37.4 minuti; i due candidati eseguiti in parallelo. Checkpoint ripresi: 0/9.
Record: `results/experiments/20261007T135726798413Z_phase37-phase14-random-ratio5-depth5-features128-200trees.json`.

## Rapporto 7:1

| Fold | F1 | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.436101368 | -0.002925479 | 0.267288788 |
| 2 | 0.440231866 | -0.003821655 | 0.287927505 |
| 3 | 0.443332291 | -0.000179355 | 0.286280142 |

Fold favorevoli: 0/3. Intervallo condizionale del delta F1: [-0.004093154, -0.000610538].
Training esterni conservati: 123624, 123624, 123632; tutti i positivi. Runner completo: 47.2 minuti; i due candidati eseguiti in parallelo. Checkpoint ripresi: 0/9.
Record: `results/experiments/20261007T140713077912Z_phase37-phase14-random-ratio7-depth5-features128-200trees.json`.

Verifiche: replay dei tre modelli del controllo; replay di tutti i 18 modelli nuovi e di tutte le probabilità OOF interne; ricostruzione delle soglie dei bin dalle righe conservate; riproduzione delle soglie interne e delle metriche; tutti i positivi conservati e tutte le validazioni escluse dai training; campioni 5:1 contenuti nei 7:1 in tutti i nove fit. Snapshot e checkpoint ogni 25 alberi conservati localmente.

Un solo seed di campionamento e rapporti fissati prima dei risultati. I fold development sono già stati consultati per varie scelte. Il bootstrap condiziona sui fit, sui campioni e sulle soglie, senza includere l'incertezza da riaddestramento o selezione ripetuta. Le probabilità grezze riflettono la prevalenza modificata dal campionamento. L'attuale configurazione corretta Phase 34 e la submission non sono modificate da questi test storici.

Riproduzione: `python -m tools.phase37_undersampling_ratios prepare`, poi `python -m tools.phase37_undersampling_ratios run`, infine `python -m tools.phase37_undersampling_ratios report --publish`. Richiede i dati e lo snapshot storico locale di Phase 35.

## Decisione

Manteniamo tutte le righe nel confronto storico. Il 7:1 riduce il calo rispetto al 5:1, ma entrambi hanno F1 inferiore al controllo in tutti i fold. Gli intervalli condizionali del delta rispetto al controllo sono interamente negativi; restano esclusi la variabilità di altri campioni, il riaddestramento e la selezione ripetuta sui fold development. Il miglior F1 nested storico resta 0,442155227.

Passati 20 test unitari selezionati. Per ciascun rapporto sono passati il workflow sintetico completo, il replay dei risultati salvati e la ripresa esatta da 25 alberi con riuso degli altri otto checkpoint completi. Anche tutti i 18 modelli reali e tutti i 18 insiemi di bin sono stati verificati. I 18 fit reali sono stati freschi, senza riprese.
