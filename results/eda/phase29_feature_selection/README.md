# Phase 29 — Selezione di 100 e 60 feature

La selezione di 100 e 60 feature è conclusa. Il riferimento Phase 14, con
295 feature e F1 annidato **0.442155227**, rimane il modello selezionato.

| Input | F1 annidato | Delta rispetto a Phase 14 | Fold favorevoli |
| ---: | ---: | ---: | ---: |
| 100 | 0.439223556 | −0.002931671 | 0/3 |
| 60 | 0.437584073 | −0.004571154 | 0/3 |

Entrambe le riduzioni perdono F1 in tutti i fold. Le tre liste esterne hanno
37 feature comuni su 100 e 19 su 60: la selezione varia sensibilmente tra i training.
Il [rapporto completo](../phase29_feature_selection_summary.md) e il
[confronto JSON](comparison.json) raccolgono metriche e verifiche.

## Metodo

La CV usa gli stessi esempi development e seed di Phase 14, con tre fold
esterni e due interni. Ciascuno dei nove training costruisce la propria selezione:

- Il selettore e il suo preprocessing sono appresi sull'80% del training;
  il restante 20% serve alla scelta della soglia e all'importanza per permutazione.
- Le feature sono ordinate per calo medio F1 su fino a 10.000 righe stratificate,
  con tre permutazioni per colonna, compresi i mancanti, e soglia fissa.
- Top 60 è un sottoinsieme di top 100. I booster ridotti sono riaddestrati
  sull'intero training con i parametri Phase 14. Il campionamento conserva
  la frazione 128/295: 44 candidate per nodo con 100 input, 27 con 60.
- Le soglie finali usano gli OOF interni. Una decima selezione sul development
  produce le liste diagnostiche consultabili qui.

## File

Le cartelle `outer_N`, `outer_N_inner_M` e `development` contengono classifiche,
liste e importanza dei gruppi. Per orientarsi:

- [Classifica development](development/ranking.csv), [top 100](development/top100.json)
  e [top 60](development/top60.json).
- [Frequenze di selezione](selection_frequency.csv).

I record sono in `results/experiments/phase29_feature_selection/`.
Checkpoint e OOF sono artefatti locali sotto
`results/boosting_artifacts/phase29_phase14_feature_selection/`, escluso da Git.

## Riproduzione

Il runner storico è `tools/phase29_feature_selection.py`, configurato da
`configs/experiments/phase29_phase14_feature_selection.json`. Richiede i quattro
CSV ufficiali, le dipendenze del progetto, gli OOF e checkpoint Phase 14,
i manifest di controllo e i sorgenti con gli hash richiesti dal preflight.
OOF, checkpoint e manifest sono artefatti locali dell'esecuzione originale.

Dalla radice del progetto, preparare le cache e avviare la prova:

```powershell
python -m tools.prepare_data
python -m tools.phase29_feature_selection preflight
python -m tools.phase29_feature_selection run
```

Per riprendere un'esecuzione già iniziata, usare soltanto `run`: il runner
conserva il preflight originale e verifica i fit completati prima del riuso.
