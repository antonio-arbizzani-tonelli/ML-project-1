# Stato finale del modello e studi conclusi

La ricerca dei modelli è **conclusa per questa versione**. Il modello adottato
è Phase 14; le prove successive non hanno fornito un miglioramento F1 che ne
motivasse la sostituzione. Questo documento riassume lo stato congelato e le
idee rinviate; il percorso dei risultati è in [EXPERIMENTS.md](EXPERIMENTS.md)
e la spiegazione della scelta è nel [report finale](FINAL_REPORT.md).

## Versione congelata

| Componente | Stato adottato |
| --- | --- |
| Modello | Gradient boosting a istogrammi implementato in NumPy |
| Input | 295 feature Phase 14, incluse HAREHAB1 e ALCDAY5 |
| Preprocessing | Non risposte documentate → NaN, zeri documentati → 0, bin a quantili appresi sul training |
| Parametri | 200 alberi, profondità 5, learning rate 0.05, minimo foglia 200, 64 bin, 128 candidate per nodo |
| Regolarizzazione / pesatura | L2 delle foglie 1; peso dei positivi 1 |
| Seed del modello | 20260920 |
| F1 development annidato | 0.442155227 |
| Precision / recall | 0.367632339 / 0.554572908 |
| Soglia finale | 0.21086909970855153, media delle tre soglie Phase 14 |

La [configurazione finale](../configs/final_model.json) e il
[piano di preprocessing](../configs/experiments/phase14_codebook_corrections_only.json)
definiscono il percorso usato da `run.py`. Il modello completo è già stato
addestrato su 328.135 righe e ha generato 109.379 predizioni test.

HAREHAB1 rimane nella versione finale. La domanda riguarda riabilitazione
post-infarto e può rivelare informazioni attraverso l'eleggibilità alla
risposta; l'ablazione isolata ha F1 0.431039. Il report documenta questo legame
con la classificazione della malattia dichiarata.

## Studi completati

| Famiglia di studio | Risultato essenziale | Evidenza |
| --- | --- | --- |
| Modelli lineari e rappresentazioni | Logistica storica annidata 0.425628; migliore ridge 0.419508 | [Phase 7](../results/eda/phase7_nested_and_ridge_summary.md) |
| Griglia del booster e correzioni semantiche | Selezione 200 alberi/profondità 5; Phase 14 passa da 0.441741 a 0.442155 | [Phase 14](../results/eda/phase14_codebook_corrections_summary.md) |
| Ablazione HAREHAB1 | 0.431039, delta −0.011116; feature conservata | [Record Phase 15](../results/experiments/20260921T154905753681Z_phase15-boosting-ablate-harehab1-depth5-features128-200trees.json) |
| Frequenze, attività e ALCDAY5 | Migliore candidato della famiglia 0.441862; nessuno adottato | [Storico](EXPERIMENTS.md#completed-comparisons-against-phase-14) |
| Binning generale e clinico | 0.440783 e 0.439731; quantili Phase 14 conservati | [Phase 20](../results/eda/phase20_binning_summary.md), [Phase 21–22](../results/eda/phase21_22_preprocessing_summary.md) |
| Indicatori e imputazione | Indicatori 0.440322; conferma annidata dell'imputazione ristretta 0.441791 | [Phase 23](../results/eda/phase23_missing_reason_flags_summary.md), [Phase 24](../results/eda/phase24_original_restricted_imputation_summary.md) |
| Parametri, profondità e peso positivo | Migliore candidato della famiglia 0.441393; configurazione Phase 14 conservata | [Storico](EXPERIMENTS.md#completed-comparisons-against-phase-14) |
| Secondo seed e media | 0.439924 e 0.441111; entrambi inferiori nei tre fold | [Phase 26](../results/eda/phase26_two_seeds_summary.md) |
| Selezione di 100/60 feature | 0.439224 e 0.437584; entrambe inferiori nei tre fold | [Phase 29](../results/eda/phase29_feature_selection_summary.md) |
| MLP con preprocessing dedicato | 64 → 32: 0.424451; 128 → 64: 0.419268; entrambe inferiori nei tre fold | [Phase 32](../results/eda/phase32_mlp_summary.md) |

Gli F1 della tabella usano la selezione annidata della soglia. Le famiglie
raggruppate riportano il migliore dei candidati effettivamente confrontati
con Phase 14, non un risultato comune a ogni variante.

Lo screening dell'imputazione Phase 16 aveva F1 esplorativo 0.442674:
selezionava la soglia sugli stessi OOF poi valutati. La conferma Phase 24,
con soglie interne, non ha confermato il vantaggio. Le interazioni elencate
nella configurazione del ramo Phase 16 non erano applicate dal preprocessing
ad alberi; il suo risultato non misura il loro effetto. I risultati lineari
storici precedono la correzione da mediana a moda del fill categoriale.

## Idee rinviate

Le seguenti possibilità restano ipotesi per un eventuale studio successivo;
non sono componenti della versione finale né una coda di training attiva:

- combinare imputazione ristretta e binning che preservi tutti gli indicatori;
- provare rappresentazioni di frequenze di esercizio o attività diverse dalle
  varianti già misurate;
- implementare interazioni effettive sul percorso ad alberi o un ensemble
  booster–logistica, con scelte apprese esclusivamente nei training interni.

Le prove concluse restano documentate nei rapporti e nei tre registri degli
esperimenti. Qualunque studio futuro costituirebbe un confronto distinto,
realizzato con NumPy e la libreria standard. Le CV development hanno già
contribuito a molte scelte e il 20% iniziale è stato consultato dalle baseline;
il risultato corrente è quindi evidenza development dopo selezione.
