# Phase 25 — L2 delle foglie da 1 a 5

## Prova e protocollo

Una sola modifica a Phase 14: l2_regularization=5 anziché 1. Stesse 295 feature, incluso HAREHAB1, preprocessing codebook, binning a quantili e gestione dei NaN. Modello: 200 alberi, profondità 5, learning rate 0,05, minimo foglia 200, 64 bin e 128 feature candidate. Seed booster 20260920.

262.508 righe development; split esterni seed 20260919 e due fold interni per esterno, con seed 20260922 + fold. Nove nuovi training del candidato. Ogni soglia è scelta solo negli OOF interni. Il controllo usa gli artefatti Phase 14 verificati, senza nuovo training della baseline.

## Risultati annidati

| Metrica | Phase 14 | L2=5 | Differenza |
| --- | ---: | ---: | ---: |
| f1 | 0.442155227 | 0.441393484 | -0.000761743 |
| precision | 0.367632339 | 0.362086120 | -0.005546219 |
| recall | 0.554572908 | 0.565185505 | +0.010612597 |
| average_precision | 0.428609250 | 0.429316126 | +0.000706876 |
| log_loss | 0.215225071 | 0.215175679 | -0.000049392 |

## Fold e soglie

| Fold | F1 Phase 14 | F1 L2=5 | Delta | Soglia Phase 14 | Soglia L2=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.439026847 | 0.437774746 | -0.001252101 | 0.203056689 | 0.203773579 |
| 2 | 0.444053521 | 0.441669145 | -0.002384376 | 0.220104637 | 0.199948567 |
| 3 | 0.443511646 | 0.444799670 | +0.001288024 | 0.209445973 | 0.210708616 |

## Matrice di confusione

| Esito | Phase 14 | L2=5 | Differenza |
| --- | ---: | ---: | ---: |
| true_positives | 12855 | 13101 | +246 |
| false_negatives | 10325 | 10079 | -246 |
| false_positives | 22112 | 23081 | +969 |
| true_negatives | 217216 | 216247 | -969 |

## Interpretazione e decisione

F1 aggregato cambia di -0.000761743; fold favorevoli 1/3. Il criterio operativo richiede F1 superiore e almeno due fold favorevoli. Criterio non soddisfatto.

**L2=5 non viene adottata. Phase 14 con L2=1 e soglia di submission 0,2108690997 resta configurata.** Questa configurazione è ora una prova conclusa.

Il candidato recupera 246 positivi in più, ma aggiunge 969 falsi positivi. Il recall aumenta di 0,010613 e la precision scende di 0,005546: il saldo dell'F1 è negativo. AP e log loss migliorano leggermente, ma non confermano un vantaggio sulla metrica principale.

Intervallo descrittivo bootstrap appaiato 95% del delta F1: [-0.002479661, +0.000965945], 20.000 ripetizioni, seed 20260929. Modelli e soglie fissi; non comprende riaddestramento o selezione ripetuta dei candidati.

F1 esplorativo, con soglia scelta sugli stessi OOF esterni valutati: 0.442325175, contro 0.442077861 del controllo sotto lo stesso protocollo ottimistico. Il piccolo +0,000247 non si conferma con soglie scelte nei fold interni. La decisione usa l'F1 annidato sopra riportato. L'intervallo bootstrap comprende zero: non prova un peggioramento certo, ma manca un miglioramento coerente che giustifichi adottare il candidato.

## Verifiche, tempi e artefatti

113 test superati prima del training, inclusi i controlli che rifiutano cambiamenti aggiuntivi al preprocessing, ai seed, alla scelta delle soglie o agli altri parametri. Preflight su tutte le nove partizioni; hash degli input uguali a Phase 14 e output delle tre baseline riprodotti esattamente. Dopo training: parametri L2=5 dei checkpoint, ordine delle feature, righe e label di training verificati; output dei tre checkpoint identici agli OOF. Soglie e metriche ricostruite dai tre artefatti OOF interni con righe, fold e hash. Sorgenti e configurazioni congelate durante la prova.

Nove fit: 49.40 minuti, esclusi preflight e verifiche finali. Ambiente: Python 3.14.3, NumPy 2.4.2. Ambiente ufficiale Python 3.9 / NumPy 1.23.1 ancora da verificare.

Il 20% iniziale non è utilizzato, ma era già stato consultato storicamente; anche la selezione ripetuta sui fold development limita la stima finale. Nessuna nuova submission prodotta da questo esperimento.

- [Record](../experiments/20260930T130145988120Z_phase25-phase14-l2-5-depth5-features128-200trees.json).
- [Confronto](../experiments/phase25_l2_comparison.json).
- [Preflight](phase25_l2_preflight.json).
- Checkpoint, OOF esterni e interni, versione dell'helper e log: results/boosting_artifacts/phase25_phase14_l2_5/.

## Stato del piano

L2=5 è conclusa e scartata per l'F1. Il confronto con un secondo seed e la media dei due booster con configurazione Phase 14 e L2=1 è stato successivamente completato nella [Phase 26](phase26_two_seeds_summary.md); entrambi i candidati sono stati scartati.

Gli OOF interni di questa prova usano L2=5 e non sostituiscono gli OOF mancanti della Phase 14 con L2=1. Tutti gli artefatti precedenti sono conservati. Risultati e modifiche sono salvati localmente; nessun commit o push in questa attività.
