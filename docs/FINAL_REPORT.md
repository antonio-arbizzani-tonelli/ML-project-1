# Classificazione BRFSS: risultati e scelta del modello

Il modello selezionato è il **gradient boosting a istogrammi NumPy, Phase 14**,
con 295 feature. Ottiene F1 **0.442155** nella valutazione development con
soglie selezionate nei fold interni della cross-validation e supera tutte le
varianti successive valutate sullo stesso protocollo.

## Dati e preprocessing

Il target `_MICHD` identifica infarto miocardico o malattia coronarica dichiarati
nel questionario BRFSS 2015. Il dataset contiene 328.135 rispondenti etichettati,
321 predittori e 109.379 righe test. I positivi sono l'8,83% del training;
il 44,79% delle celle dei predittori è mancante.

Il preprocessing conserva 295 feature, escludendo identificatori, date,
variabili di disegno dell'indagine e alcune colonne ridondanti. I codici di
non risposta diventano `NaN` secondo il significato di ciascuna variabile;
i codici che indicano quantità nulle diventano zero. Per esempio,
`ALCDAY5=888` indica assenza di consumo e `PHYSHLTH=88` indica zero giorni.
Le variabili binarie vengono ricodificate usando il training.

Phase 14 corregge quattordici lacune del parsing delle non risposte.
Il booster gestisce i `NaN` scegliendo il ramo dei mancanti in ogni split e
apprende i bin a quantili sul training. La matrice finale contiene le 295
colonne sorgenti; attività nominali e frequenze a unità miste conservano la
codifica numerica della rappresentazione selezionata.

Il piano è in [preprocessing Phase 14](../configs/experiments/phase14_codebook_corrections_only.json).

## Modello e valutazione

Il booster somma alberi Newton per minimizzare la loss logistica. La
[configurazione finale](../configs/final_model.json) contiene:

| Parametro | Valore |
| --- | ---: |
| Alberi / profondità massima | 200 / 5 |
| Learning rate / minimo righe per foglia | 0.05 / 200 |
| Bin / feature candidate per nodo | 64 / 128 |
| L2 delle foglie / peso dei positivi | 1 / 1 |
| Seed del modello | 20260920 |
| Soglia finale | 0.21086909970855153 |

I confronti finali usano le stesse 262.508 righe
development e tre fold esterni. In ogni training esterno, due fold interni
selezionano la soglia F1; il modello viene poi valutato sul fold esterno escluso.
Preprocessing e statistiche vengono appresi nel training di ciascun fit.
Per il modello completo, la soglia finale è la media delle tre soglie interne
Phase 14. L'F1 development aggrega le predizioni ottenute con la soglia
selezionata per ciascun fold.

F1 è la metrica principale perché tiene conto del recupero dei positivi e dei
falsi allarmi: con questo sbilanciamento, predire sempre negativo produce
91,17% di accuracy e F1 zero.

| Modello | F1 | Precision | Recall | Average precision | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.419508 | 0.33303 | 0.56665 | 0.39028 | 0.22917 |
| Regressione logistica | 0.425628 | 0.35728 | 0.52632 | 0.38775 | 0.22198 |
| **Booster Phase 14** | **0.442155** | **0.367632** | **0.554573** | **0.428609** | **0.215225** |
| MLP 64 → 32 | 0.424451 | 0.360722 | 0.515531 | 0.387782 | 0.221674 |
| MLP 128 → 64 | 0.419268 | 0.340005 | 0.546721 | 0.379380 | 0.222710 |

I valori lineari si riferiscono alle esecuzioni storiche con imputazione a
mediana, precedenti alla correzione del fill categoriale a moda.
Phase 14 recupera 12.855 dei 23.180 positivi development e produce 22.112
falsi positivi. Gli F1 dei tre fold sono 0.439027, 0.444054 e 0.443512.

## Percorso sperimentale

Le prime prove hanno usato training brevi per confrontare rappresentazioni
e parametri dei modelli lineari. Aumentando gli aggiornamenti e introducendo
interazioni con l'età, la logistica ha raggiunto F1 0.425628.
Il booster ha poi superato questo riferimento: il confronto tra profondità
3/5, 100/200/400 alberi e 64/128 feature candidate ha selezionato 200 alberi
di profondità 5, con F1 0.441741. Le correzioni del codebook di Phase 14
hanno portato il risultato a 0.442155.

Le prove successive hanno verificato se modifiche alla rappresentazione,
all'imputazione o al modello migliorassero Phase 14. La tabella riassume
le famiglie di confronti e riporta **il miglior F1 di ciascuna famiglia, con
soglia selezionata nei fold interni**.

| Famiglia | Varianti principali | Miglior F1 |
| --- | --- | ---: |
| Rappresentazione delle feature | Frequenze alimentari giornaliere, one-hot delle attività, rimozione di `ALCDAY5` | 0.441862 |
| Binning | Stati discreti conservati su tutte le colonne a bassa cardinalità o sulle sole variabili cliniche | 0.440783 |
| Mancanti e imputazione | Indicatori di non risposta e imputazione selettiva | 0.441791 |
| Parametri e pesatura | Foglie più piccole, learning rate alternativi, 400 alberi, L2 maggiore, profondità 7, peso positivo 2 | 0.441393 |
| Seed ed ensemble | Secondo seed e media uniforme dei due booster | 0.441111 |
| Selezione delle feature | 100 o 60 colonne selezionate per permutazione nel training | 0.439224 |
| MLP | Reti 64 → 32 e 128 → 64 con preprocessing dedicato | 0.424451 |

**Nessuna di queste famiglie ha superato F1 0.442155 di Phase 14.**
L'imputazione aveva mostrato un piccolo vantaggio nello screening iniziale,
ma la conferma con soglia scelta nei fold interni ha ottenuto 0.441791.
Alcune varianti aumentano recall o average precision, aggiungendo però abbastanza
falsi positivi da ridurre F1. La riduzione delle feature ha peggiorato tutti
i fold sia con 100 colonne sia con 60.

Le MLP usano ReLU nei due strati nascosti, uscita sigmoide, BCE e Adam.
Il preprocessing dedicato applica imputazione, scaling, one-hot e conversione
delle frequenze, con statistiche apprese nel training. Le epoche sono selezionate
su una porzione interna del training, seguite dal refit completo.
Entrambe le reti hanno ottenuto F1 inferiore al booster in tutti e tre i fold;
la rete più larga ha ottenuto il risultato peggiore.

Phase 14 resta quindi la versione selezionata: conserva la rappresentazione
corretta dal codebook e il miglior F1 dei confronti completati, con un modello
da 200 alberi. Lo [storico degli esperimenti](EXPERIMENTS.md) conserva i risultati
individuali; [MLP.md](MLP.md) descrive la pipeline neurale.

## HAREHAB1

`HAREHAB1` chiede se il rispondente abbia frequentato riabilitazione ambulatoriale
dopo un infarto. La domanda dipende dalla precedente risposta a `CVDINFR4`.
Nel development, tutte le 620 righe con un codice registrato sono positive:
anche la disponibilità della risposta contiene quindi informazione legata
alla diagnosi.

La feature è presente nel modello finale. La sua rimozione isolata riduce F1
da 0.442155 a 0.431039, una perdita di 0.011116.
Questo legame contribuisce alla classificazione della malattia dichiarata e
limita l'uso del risultato come misura di previsione prima dell'evento.

L'effetto è documentato nel [record di ablazione](../results/experiments/20260921T154905753681Z_phase15-boosting-ablate-harehab1-depth5-features128-200trees.json).

## Training completo e submission

Il modello selezionato è stato addestrato su tutte le 328.135 righe etichettate.
Il run ha richiesto **18 minuti e 18 secondi** e ha prodotto 109.379 predizioni:
14.728 positive e 94.651 negative, pari al 13,47% di positivi predetti.
La soglia applicata è 0.21086909970855153.

L'F1 disponibile è quello development, 0.442155. Il punteggio della submission
sul test ufficiale resta da acquisire. I risultati del run sono nel
[riepilogo del training completo](../results/eda/final_training_summary.json).
