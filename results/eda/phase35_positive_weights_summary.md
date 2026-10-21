> Historical study evidence. The selected combined pipeline is described in the [current report](../../docs/FINAL_REPORT.md); original replay prerequisites are in the [archive note](../../docs/ARCHIVE.md).

# Phase 35 — Pesi dei positivi 1,25 e 1,5

Confronto controllato con Phase 14 e peso 2 già valutato. I due nuovi candidati usano il codice numerico storico di Phase 30, isolato dalla successiva correzione della ricerca degli split sui mancanti. Cambia soltanto il peso della loss positiva; peso negativo 1. Stesse 262.508 righe development, 295 feature, seed e parametri; tre fold esterni e due interni per ogni soglia. Nove fit per candidato.

| Peso | F1 | Precision | Recall | AP | Falsi positivi | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.442155227 | 0.367632339 | 0.554572908 | 0.428609250 | 22112 | +0.000000000 |
| 1.25 | 0.440830852 | 0.372561865 | 0.539732528 | 0.429135259 | 21070 | -0.001324375 |
| 1.5 | 0.440613223 | 0.364315727 | 0.557333909 | 0.428593410 | 22542 | -0.001542004 |
| 2 | 0.439118448 | 0.361155162 | 0.560008628 | 0.428237932 | 22962 | -0.003036779 |

## Peso 1.25

| Fold | F1 | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.436761167 | -0.002265680 | 0.252827616 |
| 2 | 0.443335634 | -0.000717887 | 0.257905940 |
| 3 | 0.442459051 | -0.001052594 | 0.258702229 |

Fold favorevoli: 0/3. Criterio operativo (F1 superiore e almeno due fold favorevoli): non soddisfatto. Intervallo bootstrap condizionale del delta: [-0.003075318; +0.000438710].

Record: `results/experiments/20261007T115927629165Z_phase35-phase14-positive-weight1p25-depth5-features128-200trees.json`.

## Peso 1.5

| Fold | F1 | Delta F1 | Soglia interna |
| --- | ---: | ---: | ---: |
| 1 | 0.437386203 | -0.001640645 | 0.277951874 |
| 2 | 0.442100867 | -0.001952654 | 0.289111002 |
| 3 | 0.442404261 | -0.001107385 | 0.279912741 |

Fold favorevoli: 0/3. Criterio operativo (F1 superiore e almeno due fold favorevoli): non soddisfatto. Intervallo bootstrap condizionale del delta: [-0.003190407; +0.000087698].

Record: `results/experiments/20261007T115928100147Z_phase35-phase14-positive-weight1p5-depth5-features128-200trees.json`.

Verifica: replay completo dei tre checkpoint storici prima delle prove; nove partizioni e bin confrontati con Phase 30; tutti i checkpoint esterni, OOF interni, soglie e metriche dei candidati riprodotti. Sorgenti eseguiti conservati in `results/boosting_artifacts/phase35_positive_weights/source_snapshot/`; checkpoint intermedi ogni 25 alberi permettono la ripresa.

I fold development sono già stati usati per più scelte; il bootstrap considera fit e soglie fissi. Queste prove riguardano la versione storica, e non stabiliscono il miglior peso per la successiva correzione degli split. Questo confronto non modifica la configurazione finale o la submission. Durante il run, Phase 34 ha aggiornato indipendentemente la configurazione finale alla versione corretta; quel file non è un input delle suite storiche qui valutate. Il relativo controllo è stato escluso dal manifest di ripresa, mantenendo invariati sorgenti numerici, input, suite e firme delle nove partizioni; i manifest precedenti sono conservati.

Riproduzione: `python -m tools.phase35_positive_weights prepare`, poi `python -m tools.phase35_positive_weights run`. Richiede i dati e lo snapshot storico Phase 30 conservati localmente.

## Interpretazione e tempi

Il peso 1,25 aumenta la precision ma recupera 344 positivi in meno, evitando 1.042 falsi positivi; il peso 1,5 recupera 64 positivi in più e aggiunge 430 falsi positivi. Entrambi peggiorano F1 in tutti e tre i fold. Gli intervalli condizionali includono zero: il calo osservato non stabilisce un effetto negativo certo nella popolazione, ma manca un vantaggio operativo rispetto al peso 1. Per questa versione storica, mantenere peso 1. Il migliore peso sotto la correzione Phase 34 richiede un confronto abbinato su quel codice.

Il runtime di circa 497 secondi nei nuovi record misura soltanto l’ultima invocazione ripresa: sette fit già completi riutilizzati, ottavo fit ripreso da 125 alberi e nono fit nuovo. Non è il costo dei nove addestramenti originali. Ci sono state più invocazioni e un intervallo di circa 90 minuti senza avanzamento registrato; anche i tempi degli alberi possono includere tale pausa. I log e i checkpoint conservano il lavoro precedente. Nessun confronto di velocità è inferito da questi tempi.

I 15 test mirati del codice storico e quattro controlli di integrazione sono passati, inclusa l’identità delle predizioni tra ripresa da 25 alberi e fit continuo a 200 alberi per entrambi i pesi. Il vecchio risultato peso 2 è stato riprodotto dagli OOF e verificato sugli stessi input, righe, fold e sorgenti. [Confronto JSON](../experiments/phase35_positive_weights_comparison.json).
