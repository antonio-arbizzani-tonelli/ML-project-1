# Verifica del binning discreto — 30 settembre 2026

## Conclusione

Il binning a quantili della Phase 14 accorpa valori che potrebbero restare distinti entro il limite di 64 bin. Il fenomeno interessa 79–82 colonne a bassa cardinalità nei training esterni, comprese età, giorni di cattiva salute, ipertensione e diabete. Quattro colonne perdono la distinzione fra tutti i loro valori osservati. Due degli otto indicatori Phase 16 ristretta vengono annullati dal binning.

La verifica conferma una perdita di risoluzione nella rappresentazione. **Non misura un miglioramento dell'F1**: conservare più dettagli può aiutare il modello oppure aumentare la variabilità delle sue stime. Per decidere occorre un confronto del booster con soglia annidata.

## Dati, metodo e integrità

- Riferimento: Phase 14, 295 feature, 64 bin; configurazione finale invariata.
- Development: 262.508 righe; il 20% di validazione iniziale non è stato utilizzato.
- Analizzati i training dei tre fold esterni e dei sei fold interni, con gli split e i seed originali: esterni `20260919`, interni `20260922 + numero del fold esterno`.
- Riprodotti preprocessing Phase 14 e `HistogramBinner` effettivi. I quantili sono calcolati esclusivamente sul training di ciascuna partizione.
- I sei hash degli input coincidono con il record Phase 14; righe, etichette e assegnazioni dei fold coincidono con l'artefatto OOF salvato.
- Per ogni colonna si confrontano valori finiti distinti e bin effettivamente occupati. Il confronto è verificato anche sulla matrice trasformata dal binner.
- La bassa cardinalità indica da 2 a 64 valori finiti distinti osservati nel training. Il criterio include anche alcune variabili dichiarate continue che nel campione hanno pochi valori distinti.
- La corrispondenza `NaN` → stato 255 è verificata su tutte le colonne; non si confonde la fusione di valori osservati con la gestione dei mancanti.
- Gli indicatori sono costruiti attraverso il percorso reale Phase 16 ristretta con `fill_values=False`, applicato alle matrici sorgente Phase 14 invariate. Nessun imputatore predittivo è eseguito.
- Il one-hot delle attività è una simulazione diagnostica con categorie del solo training e `NaN` conservati. Non è stato aggiunto alla pipeline finale.
- **Modelli predittivi addestrati: 0. Imputatori predittivi addestrati: 0.** Sono state apprese soltanto trasformazioni dei dati.
- Ambiente effettivo: Python 3.14.3, NumPy 2.4.2; tempo della verifica: 74,1 secondi. Non è una verifica nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1.

## Risultato complessivo

| Training | Righe | Colonne con 2–64 valori | Colonne che accorpano valori | Colonne con tutti i valori osservati accorpati |
| --- | ---: | ---: | ---: | ---: |
| Fold esterno 1 | 175.005 | 245 | 79 | 4 |
| Fold esterno 2 | 175.005 | 246 | 82 | 4 |
| Fold esterno 3 | 175.006 | 245 | 79 | 4 |

75 colonne risultano coinvolte in tutti e tre i training esterni; l'unione comprende 83 colonne. Nei sei training interni le colonne coinvolte sono rispettivamente 77, 76, 79, 76, 78 e 77. Le stesse quattro colonne perdono tutta la distinzione fra valori osservati in tutte e nove le partizioni.

Nel primo training esterno, le 79 colonne coinvolte sono 31 categoriche, 29 conteggi, 13 ordinali, 2 binarie e 4 dichiarate continue. Questi numeri descrivono le colonne, non il numero di soggetti danneggiati o di errori di classificazione.

## Esempi rilevanti della rappresentazione attuale

| Feature | Valori distinti nei tre training esterni | Bin occupati nei tre training esterni | Osservazione |
| --- | --- | --- | --- |
| `_AGE80` | 63 / 63 / 63 | 52 / 52 / 52 | Alcune età sono accorpate pur avendo capacità per tutte le 63 modalità. |
| `PHYSHLTH` | 31 / 31 / 31 | 12 / 13 / 12 | Parte dei conteggi da 0 a 30 giorni diventa indistinguibile. |
| `MENTHLTH` | 31 / 31 / 31 | 11 / 11 / 11 | Analoga perdita di risoluzione sui giorni. |
| `BPHIGH4` | 4 / 4 / 4 | 3 / 3 / 3 | I codici 3 e 4 diventano indistinguibili. |
| `DIABETE3` | 4 / 4 / 4 | 3 / 3 / 3 | I codici 1 e 2 diventano indistinguibili. |
| `ALCDAY5` | 38 / 38 / 37 | 18 / 18 / 18 | Molte frequenze sono accorpate. |
| `DROCDY3_` | 35 / 35 / 34 | 18 / 17 / 18 | Anche il derivato uniforme perde modalità. |
| `_STATE` | 53 / 53 / 53 | 47 / 47 / 47 | Alcuni codici territoriali sono accorpati numericamente. |

Nel primo fold:

- `BPHIGH4`: 101.182 risposte «No» (codice 3) condividono il bin con 1.722 risposte «borderline/pre-ipertensione» (codice 4). I positivi `_MICHD` sono rispettivamente 3.803 e 112: è un confronto descrittivo, non una prova dell'utilità della separazione.
- `DIABETE3`: 22.564 risposte «Yes» (codice 1) condividono il bin con 1.419 risposte «Yes, only during pregnancy» (codice 2). I positivi sono rispettivamente 4.981 e 62. Il significato del codice 2 è verificato nel codebook, poiché manca nel registro JSON.
- `_AGE80`: sono accorpate anche le età 78 e 79, oltre a diverse coppie di età più giovani. Il dettaglio di tutte le fusioni è nel JSON.

`GENHLTH` conserva i suoi cinque valori, `INCOME2` gli otto valori, e `HAREHAB1` i suoi due valori osservati. Il problema non riguarda ogni feature.

## Le quattro colonne con tutti i valori osservati accorpati

| Feature | Valori dopo il preprocessing | Righe della modalità meno frequente nei tre training esterni |
| --- | --- | --- |
| `PVTRESD1` | 0 e 1 | 11 / 20 / 15 |
| `PVTRESD2` | 0 e 1 | 436 / 405 / 447 |
| `_FRUITEX` | 0 e 2 | 35 / 34 / 37 |
| `_VEGETEX` | 0 e 2 | 18 / 15 / 21 |

Le risposte osservate diventano indistinguibili, ma i `NaN` restano separati: queste colonne possono ancora fornire informazione attraverso la presenza o assenza di risposta. Le modalità perse sono rare; la loro conservazione da sola non implica un grande guadagno di F1.

## Indicatori dei motivi dei mancanti

Il percorso Phase 16 ristretta produce otto indicatori. In tutti e nove i training, sei conservano i due stati e due diventano costanti dopo il binning:

| Indicatore annullato | Righe con valore 1 nei tre training esterni | Frequenza sulle righe del training |
| --- | --- | --- |
| `EMPLOY1__was_refused` | 1.475 / 1.517 / 1.508 | 0,843–0,867% |
| `PNEUVAC3__was_refused` | 856 / 797 / 819 | 0,455–0,489% |

Conservano i due stati: i due indicatori di reddito, `PNEUVAC3__was_unknown`, i mancanti di altezza e peso e il mancante di BMI. Il controllo identifica un limite dei confronti sugli indicatori effettuati con il binner attuale; non quantifica il contributo dei due stati persi all'F1.

## Conseguenze sul futuro one-hot delle attività

Le colonne sorgente `EXRACT11` hanno 75 categorie in ciascun training esterno e diventano 20 bin. `EXRACT21` ha 76 categorie e diventa 25, 25 e 24 bin. Sono oltre la soglia di 64 modalità della proposta iniziale, quindi una regola esatta soltanto sotto tale soglia non risolve direttamente la loro rappresentazione nominale.

Applicando il binner attuale ai singoli indicatori one-hot:

| Training | `EXRACT11`: indicatori annullati / totali | `EXRACT21`: indicatori annullati / totali |
| --- | ---: | ---: |
| Fold esterno 1 | 64 / 75 | 63 / 76 |
| Fold esterno 2 | 64 / 75 | 62 / 76 |
| Fold esterno 3 | 64 / 75 | 63 / 76 |

Nel primo training, gli indicatori annullati corrispondono rispettivamente a 15.508 e 18.581 risposte osservate. Sono conteggi riferiti alle due feature, non soggetti unici. Fra gli indicatori annullati, 22 e 23 hanno almeno 200 risposte positive; gli altri sono più rari. La conservazione dei loro stati non garantisce che il booster sappia utilizzarli: contano anche il minimo di 200 righe per foglia, il trattamento dei mancanti e il campionamento delle feature.

## Proposta al termine dell'audit

I passaggi seguenti descrivono la proposta formulata al termine di questo audit, prima dei confronti di training successivi.

1. Dare priorità a un confronto isolato del binning: preservare esattamente i valori distinti quando sono al massimo 64, mantenere i quantili per le altre colonne e mantenere lo stato separato dei `NaN`.
2. Conservare tutte le 295 feature e gli altri parametri Phase 14; scegliere nuovamente le soglie nei fold interni. Il confronto misurerà se la maggiore risoluzione migliora l'F1 aggregato e con quale coerenza tra i fold.
3. Verificare le categorie mai osservate nel training e le colonne che passano da 64 a 65 modalità tra partizioni. Per i valori ordinati, la regola di assegnazione fuori dai valori osservati deve essere deterministica.
4. Dopo tale confronto, valutare i motivi dei mancanti e le categorie nominali su un controllo corrispondente. Se si aggiunge one-hot o indicatori, i loro stati binari devono sopravvivere al binning.

Durante questo audit non era stato modificato il booster né avviato il confronto di training. Il binning esatto è stato poi valutato nelle [Phase 20](phase20_binning_summary.md) e [Phase 21](phase21_preprocessing_summary.md), mentre one-hot delle attività e indicatori di mancata risposta sono stati esaminati nelle [Phase 22](phase22_preprocessing_summary.md) e [Phase 23](phase23_missing_reason_flags_summary.md). Queste varianti non sono state adottate. La Phase 14 resta il riferimento con F1 annidato 0,44216.

## Artefatti

- [Risultati completi e hash](discrete_binning_audit.json).
- [Tabella delle 295 feature in tutte le nove partizioni](discrete_binning_features.csv).
- [Utilità per riprodurre la verifica](../../tools/audit_discrete_binning.py).
- [Implementazione analizzata](../../src/numpy_boosting.py).
- [Codebook per il significato di DIABETE3](../../official_material/project1/brfss_2015_codebook.txt).
