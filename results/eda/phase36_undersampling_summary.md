> Historical study evidence. The selected combined pipeline is described in the [current report](../../docs/FINAL_REPORT.md); original replay prerequisites are in the [archive note](../../docs/ARCHIVE.md).

# Phase 36 — Due metodi di undersampling

Confronto con la versione corretta Phase 34. Tutti i positivi restano nel training; si conservano tre negativi per positivo. Random sceglie i negativi uniformemente. Hybrid hard conserva metà negativi casuali e metà tra quelli con probabilità positiva maggiore secondo un pilot di 24 alberi, profondità 3, addestrato esclusivamente sul training del fold. Il pilot usa a sua volta un campione casuale 3:1.

Stesse 295 colonne, parametri del modello finale e seed del controllo. Campionamento indipendente nei tre fold esterni e nei sei interni. Preprocessing finale e bin apprendono solo dalle righe conservate. Validazione esterna e interna mantengono tutte le righe e la prevalenza originale. Soglie F1 scelte tramite OOF interno; holdout e test non consultati.

| Metodo | F1 | Precision | Recall | AP | Falsi positivi | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tutti i dati | 0.440803240 | 0.362001109 | 0.563459879 | 0.428781433 | 23019 | +0.000000000 |
| random | 0.439388760 | 0.363860265 | 0.554486626 | 0.425835951 | 22471 | -0.001414480 |
| hybrid_hard | 0.393534831 | 0.314654950 | 0.525194133 | 0.384225019 | 26516 | -0.047268409 |

## random

| Fold | F1 | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.435385160 | -0.001336289 | 0.471973692 |
| 2 | 0.441968912 | +0.000233812 | 0.481669488 |
| 3 | 0.440899501 | -0.002833570 | 0.479532323 |

Fold favorevoli: 1/3. Intervallo bootstrap condizionale del delta F1: [-0.003355609, +0.000540069].
Tempo totale dei nove fit, selezione/pilot inclusi: 1456.7 secondi (somma dei tempi per fit, due candidati eseguiti in parallelo).
Righe conservate nei fold esterni: 61812, 61812, 61816; positivi tutti conservati. Checkpoint ripresi: 0/9.
Record: `results/experiments/20261007T124912476309Z_phase36-undersampling-random-depth5-features128-200trees.json`.

## hybrid_hard

| Fold | F1 | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.387837389 | -0.048884060 | 0.305151003 |
| 2 | 0.397755028 | -0.043980072 | 0.319127579 |
| 3 | 0.395340202 | -0.048392868 | 0.306946255 |

Fold favorevoli: 0/3. Intervallo bootstrap condizionale del delta F1: [-0.051378719, -0.043164603].
Tempo totale dei nove fit, selezione/pilot inclusi: 1650.7 secondi (somma dei tempi per fit, due candidati eseguiti in parallelo).
Righe conservate nei fold esterni: 61812, 61812, 61816; positivi tutti conservati. Checkpoint ripresi: 0/9.
Record: `results/experiments/20261007T125233036668Z_phase36-undersampling-hybrid-hard-depth5-features128-200trees.json`.

Verifiche: replay dei tre checkpoint del controllo; riproduzione esatta di tutte le selezioni; controllo dei confini di tutti i nove training; replay dei tre checkpoint finali e dei tre pilot esterni; OOF interni, soglie e metriche riprodotti. Snapshot sorgenti e checkpoint ogni 25 alberi conservati localmente.

I fold development sono già stati usati per varie scelte. Gli intervalli condizionano sui fit e sulle soglie; non includono variabilità del campionamento, riaddestramento o selezione ripetuta. Rapporto 3:1 e pilot fissati prima dei risultati: nessuna ricerca di iperparametri. L'undersampling cambia la prevalenza e, soprattutto con selezione hard, la distribuzione dei negativi: le probabilità grezze non sono calibrate sulla popolazione originale.

Riproduzione: `python -m tools.phase36_undersampling prepare`, poi `python -m tools.phase36_undersampling run`, infine `python -m tools.phase36_undersampling report --publish`. Configurazione finale e submission non modificate.

## Decisione e verifiche aggiuntive

Manteniamo tutte le righe con la versione corretta Phase 34. Il casuale è vicino al controllo ma non migliora F1; il metodo mirato peggiora F1 in tutti i fold e riduce anche AP, precision e recall. Queste conclusioni riguardano i due metodi 3:1 fissati, con un solo seed di campionamento.

Replay aggiuntivo riuscito per tutti i 18 modelli principali, tutti i 9 pilot e tutti i 18 insiemi di soglie dei bin, ricostruiti dalle sole righe conservate. Riprodotte esattamente anche le probabilità OOF interne. Passati 17 test unitari e, per ciascun metodo, il workflow sintetico completo e la ripresa da 25 alberi con riuso degli altri otto checkpoint completi.

Tempi dell’intero runner (preprocessing, selezione/pilot, fit e previsioni): casuale 24.9 minuti, mirato 28.2 minuti. I due candidati sono stati eseguiti contemporaneamente, senza riprese. La somma dei tempi per fit riportata sopra esclude alcuni passaggi di preprocessing del runner. Il controllo storico aveva due modelli esterni ripresi da 125 alberi e timer con pause: non è un riferimento pulito per quantificare il fattore di accelerazione.
