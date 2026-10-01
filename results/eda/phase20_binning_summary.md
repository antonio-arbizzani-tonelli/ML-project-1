# Phase 20 — binning esatto delle colonne con poche modalità

## Protocollo e controlli

Confronto isolato con Phase 14: stesse 295 feature, stessa pulizia dei codici, stessi parametri del booster e stessi split. L'unica opzione aggiunta è `binning_strategy=exact_low_cardinality`. Fino a 64 valori finiti distinti nel training si conservano tutte le modalità; oltre 64 si riusa il binning a quantili. I NaN restano nel bin 255.

Per le colonne esatte si valuta anche la divisione fra tutti gli osservati e tutti i mancanti: senza questo confine, eliminare i bin vuoti potrebbe ridurre la capacità precedente del modello. Le colonne che restano a quantili mantengono la ricerca dei confini originale.

Validazione sui 262.508 esempi development: tre fold esterni e due interni per scegliere ciascuna soglia. Nove fit completi da 200 alberi; Phase 14 riutilizzata senza riaddestramento. I test unitari addestrano piccoli modelli sintetici. Il primo avvio della CV, interrotto per un problema di logging prima del completamento del primo fit, non contribuisce ai risultati e al runtime del record.

95 test passati prima del training. Il preflight sui nove training conferma tutte le modalità esatte, il fallback a quantili identico e lo stato dei mancanti invariato. I tre checkpoint Phase 14 ricalcolano tutte le predizioni salvate con differenza massima zero. I profili dei nove training reali coincidono con i conteggi dei bin del preflight.

Hash degli input, righe, etichette e fold OOF verificati; predizioni annidate e metriche ricalcolate. Anche le predizioni dei tre checkpoint candidati sono ricalcolate, con differenza massima zero dagli OOF. Gli hash delle righe e delle etichette usate per il training coincidono con Phase 14. Il 20% iniziale non è utilizzato in questa prova; era già stato consultato storicamente.

## Metriche annidate aggregate

| Modello | F1 | Precision | Recall | AP | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 14 | 0.442155 | 0.367632 | 0.554573 | 0.428609 | 0.215225 |
| Binning esatto | 0.440783 | 0.368551 | 0.548231 | 0.428874 | 0.215222 |

Differenza F1 assoluta: **-0.001372**. Fold favorevoli: **1/3**.

## Risultati per fold

| Fold | F1 Phase 14 | F1 candidato | Differenza | Soglia Phase 14 | Soglia candidato |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.439027 | 0.437242 | -0.001785 | 0.203057 | 0.217115 |
| 2 | 0.444054 | 0.440956 | -0.003097 | 0.220105 | 0.211153 |
| 3 | 0.443512 | 0.444055 | +0.000544 | 0.209446 | 0.210802 |

## Matrici di confusione

| Modello | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 | 12855 | 10325 | 22112 | 217216 |
| Binning esatto | 12708 | 10472 | 21773 | 217555 |

## Cambiamenti nelle decisioni

Il candidato recupera 327 falsi negativi e perde 474 veri positivi. Introduce 1187 nuovi falsi positivi e ne elimina 1526.

Intervallo bootstrap appaiato condizionato al 95% della differenza F1: [-0.003058, +0.000288]. Usa 20.000 ricampionamenti stratificati per classe e seed 20260929, a modelli e soglie fissati. Non comprende la variabilità del training o la selezione ripetuta dei candidati: è un indicatore diagnostico, non una conferma indipendente.

## Costo e ambiente

Runtime della CV completata: 2523.0 secondi (42.0 minuti). Ambiente effettivo: Python 3.14.3, NumPy 2.4.2. Non è stato eseguito il controllo nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1.

Il candidato usa 250, 251 e 250 colonne esatte nei tre training esterni, includendo le colonne costanti. Le colonne che recuperano distinzioni sono 79, 82 e 79. Nei training interni si usano 251 colonne esatte. La matrice resta uint8; il numero maggiore di confini può aumentare il costo della ricerca degli split.

## Decisione

Il candidato non soddisfa la regola operativa del piano. Conservare Phase 14 come riferimento; questo confronto non supporta l'adozione generale del binning esatto.

La variazione AP è +0.000265 e quella della log loss è -0.000003: sono differenze molto piccole. La correlazione delle probabilità è 0.996414. Anche scegliendo esplorativamente la soglia sulle stesse etichette OOF, l'F1 del candidato è 0.441492, contro 0.442078. Questi ultimi numeri sono ottimistici e non entrano nella decisione.

La perdita di modalità è reale, ma aumentare la risoluzione di tutte le colonne coinvolte non migliora l'F1 in questa prova. Il risultato non dimostra che la causa sia l'overfitting, né esclude benefici su una famiglia specifica. Un eventuale seguito deve limitare l'esatta conservazione a binarie e categorie con poche modalità, mantenendo il binning numerico originale; nei futuri one-hot e indicatori va assicurata la conservazione dei loro due stati. L'intervallo condizionato include zero: non presentare la piccola differenza come una prova definitiva di inferiorità del metodo in generale.

La configurazione di submission resta invariata: questa richiesta implementa e valuta il candidato. Non è stato addestrato un nuovo modello sull'intero training né generata una submission.

## Artefatti

- [Record candidato](../experiments/20260930T005323542120Z_phase20-phase14-exact-low-cardinality-depth5-features128-200trees.json).
- [Confronto completo](../experiments/phase20_binning_comparison.json).
- [Preflight](phase20_binning_preflight.json).
- [Audit iniziale](discrete_binning_audit.md).
- [Configurazione](../../configs/experiments/phase20_phase14_exact_low_cardinality.json).
