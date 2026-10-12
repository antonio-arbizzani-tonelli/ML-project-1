# Classificazione BRFSS: note estese a supporto del report

Il report di consegna è [FINAL_REPORT.pdf](FINAL_REPORT.pdf), scritto in
LaTeX, in inglese, e verificato su **due pagine**, senza appendice. Il
[sorgente modificabile](FINAL_REPORT.tex) contiene la versione breve con
metodi, prove negative, risultati e limiti. Queste note conservano i dettagli
e i collegamenti alle evidenze.

Il modello selezionato è il **gradient boosting a istogrammi NumPy, Phase 14**,
con 295 feature e l'implementazione storica inclusa nel repository.
La submission finale ottiene **F1 0.452 e accuracy 0.878 su AIcrowd**,
secondo i valori comunicati dal team il 12 ottobre 2026, visualizzati a tre
decimali. La valutazione development annidata dà F1 **0.442155**; la
calibrazione OOF su tutto il train dà **0.442770**. Sono tre protocolli distinti.
Phase 34 corregge la ricerca degli split e resta un confronto documentato,
con F1 development 0.440803. Dopo aver caricato entrambe le versioni,
il team ha scelto Phase 14 in base al feedback AIcrowd; non è stato fornito
il punteggio numerico ufficiale di Phase 34.

## Dati e preprocessing

Il preprocessing di base è parte della pipeline selezionata: l'audit del
codebook guida le esclusioni, distingue codici validi e non risposte e
mantiene separati valori mancanti e quantità nulle. Le prove negative
riportate più avanti modificano questa rappresentazione già pulita.
Il loro esito riguarda quelle aggiunte, senza misurare da solo il contributo
complessivo della pulizia iniziale.

Il target `_MICHD` identifica infarto miocardico o malattia coronarica dichiarati
nel questionario BRFSS 2015. Il dataset contiene 328.135 rispondenti etichettati,
321 predittori e 109.379 righe test. I positivi sono l'8,83% del training;
il 44,79% delle celle dei predittori è mancante.

Il preprocessing conserva 295 feature, escludendo identificatori, data completa di intervista,
variabili di disegno dell'indagine e alcune colonne ridondanti. I codici di
non risposta diventano `NaN` secondo il significato di ciascuna variabile;
i codici che indicano quantità nulle diventano zero. Per esempio,
`ALCDAY5=888` indica assenza di consumo e `PHYSHLTH=88` indica zero giorni.
Le variabili binarie vengono ricodificate usando il training.

Sulle 295 colonne development, la pulizia converte 1.095.421 codici numerici
di non risposta in `NaN` e 899.687 codici di quantità nulla in zero.
Questi [conteggi](../results/eda/phase33_preprocessing_counts.json) descrivono
il lavoro sulla rappresentazione.

Il [benchmark completo sui dati grezzi](../results/eda/phase33_raw_preprocessing_summary.md),
eseguito prima della correzione dell'algoritmo,
mantiene modello e fold invariati, conservando tutti i 321 predittori,
i codici numerici originali e i `NaN`. Ottiene F1 **0.441457**, rispetto a
**0.442155** dopo la pulizia: **+0.000698**, con un piccolo vantaggio in
ciascuno dei tre fold. L'intervallo appaiato al 95% per pulito meno grezzo
è **[−0.001226, +0.002635]**, condizionato a fit e soglie fissi; include zero.
La differenza predittiva è contenuta e incerta. Il confronto misura
ricodifiche ed esclusioni insieme, con 128 feature candidate su 321 oppure
295 colonne; le singole operazioni restano da isolare.

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
| Soglia finale | 0.2106169795070252 |

I confronti finali usano le stesse 262.508 righe
development e tre fold esterni. In ogni training esterno, due fold interni
selezionano la soglia F1; il modello viene poi valutato sul fold esterno escluso.
Preprocessing e statistiche vengono appresi nel training di ciascun fit.
L'F1 development aggrega le predizioni ottenute con la soglia selezionata
per ciascun fold. Per la submission finale, tre nuovi fold stratificati
coprono tutte le 328.135 righe train: preprocessing e booster sono appresi
nel training di ciascun fold. La soglia massimizza F1 sulle predizioni OOF
aggregate, senza usare etichette test. L'F1 di calibrazione descrive le stesse
etichette che hanno selezionato la soglia e non è una valutazione indipendente.

F1 è la metrica principale perché tiene conto del recupero dei positivi e dei
falsi allarmi: con questo sbilanciamento, predire sempre negativo produce
91,17% di accuracy e F1 zero.

| Modello | F1 | Precision | Recall | Average precision | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.419508 | 0.33303 | 0.56665 | 0.39028 | 0.22917 |
| Regressione logistica | 0.425628 | 0.35728 | 0.52632 | 0.38775 | 0.22198 |
| Booster sui 321 predittori grezzi | 0.441457 | 0.366631 | 0.554659 | 0.427835 | 0.215348 |
| **Booster Phase 14 selezionato** | **0.442155** | **0.367632** | **0.554573** | **0.428609** | **0.215225** |
| Booster Phase 34, ricerca corretta | 0.440803 | 0.362001 | 0.563460 | 0.428781 | 0.215229 |
| MLP 64 → 32 | 0.424451 | 0.360722 | 0.515531 | 0.387782 | 0.221674 |
| MLP 128 → 64 | 0.419268 | 0.340005 | 0.546721 | 0.379380 | 0.222710 |

I valori lineari si riferiscono alle esecuzioni storiche con imputazione a
mediana, precedenti alla correzione del fill categoriale a moda.
Il Phase 14 selezionato recupera 12.855 dei 23.180 positivi development e produce 22.112
falsi positivi. Gli F1 dei tre fold sono 0.439027, 0.444054 e 0.443512.

## Percorso sperimentale

Le prime prove hanno usato training brevi per confrontare rappresentazioni
e parametri dei modelli lineari. Aumentando gli aggiornamenti e introducendo
interazioni con l'età, la logistica ha raggiunto F1 0.425628.
Il booster ha poi superato questo riferimento: il confronto tra profondità
3/5, 100/200/400 alberi e 64/128 feature candidate ha selezionato 200 alberi
di profondità 5, con F1 0.441741. Le correzioni del codebook di Phase 14
hanno portato il risultato a 0.442155. Il guadagno è piccolo (+0.000415)
e cambia segno tra i fold: la correttezza semantica del codebook resta la
motivazione principale per conservarle.

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

Queste prove hanno guidato la scelta del piano Phase 14. La versione finale
conserva quelle scelte e il codice storico per riprodurre la submission
migliore. Phase 34 valuta separatamente la ricerca degli split corretta;
le altre varianti storiche non sono state ripetute dopo la correzione.
Lo [storico degli esperimenti](EXPERIMENTS.md) conserva i risultati
individuali; [MLP.md](MLP.md) descrive la pipeline neurale.

## HAREHAB1

`HAREHAB1` chiede se il rispondente abbia frequentato riabilitazione ambulatoriale
dopo un infarto. La domanda dipende dalla precedente risposta a `CVDINFR4`.
Nel development, tutte le 620 righe con un codice registrato sono positive:
anche la disponibilità della risposta contiene quindi informazione legata
alla diagnosi.

La feature è presente nel modello finale. Nel controllo storico Phase 14,
la sua rimozione isolata riduce F1
da 0.442155 a 0.431039, una perdita di 0.011116.
Questo legame contribuisce alla classificazione della malattia dichiarata e
limita l'uso del risultato come misura di previsione prima dell'evento.

L'effetto è documentato nel [record di ablazione](../results/experiments/20260921T154905753681Z_phase15-boosting-ablate-harehab1-depth5-features128-200trees.json).

## Verifica della ricerca degli split: Phase 34

La revisione del booster ha trovato un'omissione nella ricerca degli split
con bin a quantili: quando i bin estremi contengono valori osservati, il
modello poteva saltare la separazione fra tutti gli osservati e tutti i
`NaN`. La correzione aggiunge il confine finale quando il nodo contiene
mancanti, mantenendo i vincoli su dimensione delle foglie e guadagno minimo.
È applicata alle strategie recursive e levelwise.

Sono passati 49 test sul booster, inclusi otto test di regressione per questa
correzione. Una verifica indipendente su 700 casi confronta gli split con
l'enumerazione completa dei candidati e non trova omissioni.
Gradienti, Hessiane e aggiornamenti delle foglie restano quelli della loss
logistica con regolarizzazione L2.

Il confronto Phase 34 mantiene le stesse 295 colonne, gli stessi parametri
e i tre fold esterni con due fold interni. Tutti i nove modelli della versione
corretta vengono riaddestrati; il controllo usa i risultati storici Phase 14.
Questa prova misura l'effetto della correzione dell'algoritmo. Il confronto
grezzo/pulito Phase 33 è invece riferito alla precedente implementazione.

I tre nuovi modelli esterni danno AP 0.428781, rispetto a 0.428609, e log loss
0.215229, rispetto a 0.215225. Le probabilità hanno correlazione 0.99693.
Su HAREHAB1 e CVDASPRN, un primo bin vuoto permetteva già alla vecchia ricerca
di separare osservati e mancanti. Questo aiuta a spiegare il cambiamento
contenuto nel ranking. Nei tre modelli storici si contano 830 split che
separano osservati e mancanti; nei corretti 805, inclusi 270 al confine
finale ora disponibile. Molte di queste separazioni erano già rappresentabili.

Il risultato nested completo è F1 **0.440803**, rispetto a **0.442155**:
**−0.001352**, con due fold in calo e uno in lieve crescita. Il modello
corretto recupera 13.061 positivi e genera 23.019 falsi positivi:
206 recuperi e 907 falsi allarmi in più. L'intervallo appaiato al 95%
**[−0.003005, +0.000319]**, condizionato a modelli e soglie fissi, include zero.
La correzione completa i candidati validi; la prova mostra un effetto
predittivo piccolo e incerto. Questo risultato lascia aperto il margine
di miglioramento con altre scelte.

La [diagnostica delle soglie](../results/eda/phase34_threshold_diagnostic.json)
incrocia score e soglie delle due versioni. Con i nuovi score e le vecchie
soglie, F1 è 0.441583; con le nuove soglie è 0.440803. Anche il punto operativo
contribuisce quindi alla differenza. Queste celle sono descrittive:
le soglie del confronto development sono quelle selezionate nei fold interni,
senza scegliere in base all'F1 esterno. La successiva calibrazione finale su
tutto il train usa nuovi OOF. Il [confronto completo](../results/experiments/phase34_missing_only_split_comparison.json)
conserva metriche, transizioni e curve diagnostiche; il
[riepilogo](../results/eda/phase34_missing_only_split_summary.md) riporta la prova.

La [nuova analisi degli errori](../results/eda/phase34_error_followup.json)
trova 9.363 positivi mancati sia dal booster sia dalla logistica, il 40,39%
dei positivi development. Fra questi, 2.307 hanno età `_AGE80 >= 65` e salute
ottima o molto buona (`GENHLTH` 1/2). `_AGE80` è l'età imputata dal CDC;
questa definizione riproduce i 2.332 casi della precedente analisi.
L'unione delle decisioni recupera 756 positivi e aggiunge 3.480 falsi
positivi, con F1 0.43521. Il recupero ulteriore comporta abbastanza falsi
allarmi da peggiorare il risultato. Questi errori sono coerenti con la
difficoltà di ricostruire una diagnosi passata dalle risposte attuali;
non dimostrano un limite delle prestazioni raggiungibili.

## Ultime prove: pesi e campionamento del training

Le ultime sei configurazioni non migliorano il proprio controllo abbinato.
La versione finale Phase 14 mantiene tutte le righe e peso positivo 1.
I pesi 1,25/1,5 e i rapporti casuali 5:1/7:1 sono valutati sul
codice storico Phase 14; il rapporto 3:1 e il metodo hybrid hard usano la
versione corretta Phase 34. Questi rapporti non formano un unico confronto
fra quantità di dati a parità di algoritmo.

| Prova | Controllo | F1 | Delta F1 | Fold favorevoli |
| --- | --- | ---: | ---: | ---: |
| Peso positivo 1,25 | Phase 14 storico | 0.440831 | −0.001324 | 0/3 |
| Peso positivo 1,5 | Phase 14 storico | 0.440613 | −0.001542 | 0/3 |
| Undersampling casuale 5:1 | Phase 14 storico | 0.438226 | −0.003929 | 0/3 |
| Undersampling casuale 7:1 | Phase 14 storico | 0.439820 | −0.002336 | 0/3 |
| Undersampling casuale 3:1 | Phase 34 corretto | 0.439389 | −0.001414 | 1/3 |
| Hybrid hard 3:1 | Phase 34 corretto | 0.393535 | −0.047268 | 0/3 |

I [pesi intermedi](../results/eda/phase35_positive_weights_summary.md) perdono
F1 in tutti i fold. Peso 1,25 evita 1.042 falsi positivi e perde 344 veri
positivi; peso 1,5 recupera 64 positivi e aggiunge 430 falsi allarmi.
Gli intervalli condizionali comprendono zero. Queste prove sostengono il
peso 1 nel codice storico; il peso migliore sul codice corretto resta da
valutare con un confronto dedicato.

Il [campionamento 3:1](../results/eda/phase36_undersampling_summary.md)
mantiene tutti i positivi e sceglie tre negativi per positivo. Il metodo
ibrido combina metà negativi casuali e metà con score più alto da un pilot
di 24 alberi, profondità 3, addestrato solo nel training del fold.
Statistiche, preprocessing e bin usano le righe conservate; le validazioni
mantengono tutte le righe e la prevalenza originale. Il metodo ibrido riduce
anche AP da 0.428781 a 0.384225, con F1 inferiore in tutti i fold e intervallo
del delta interamente negativo. La minore copertura della distribuzione
dei negativi è una spiegazione plausibile di questo peggioramento.

I [rapporti casuali 5:1 e 7:1](../results/eda/phase37_undersampling_ratios_summary.md)
usano la stessa permutazione per ciascun fit: le righe 5:1 sono un sottoinsieme
di quelle 7:1. Entrambi perdono F1 in tutti i fold rispetto al controllo
storico, con intervalli condizionali interamente negativi. Il 7:1 riduce il
calo rispetto al 5:1. Manteniamo il training completo, che conserva la
copertura dei negativi. Le prove usano un solo seed di campionamento;
gli intervalli tengono fissi fit, campioni e soglie e non comprendono
variabilità da riaddestramento o selezione ripetuta.

## Training completo e submission

Il confronto finale ha eseguito tre fit OOF e un fit completo per ciascuna
implementazione, con identici fold e parametri. Phase 14 seleziona soglia
**0.2106169795070252**, con F1 di calibrazione **0.442769549**; Phase 34
seleziona **0.2214498433950319**, con F1 di calibrazione **0.443569854**.
Il vantaggio OOF di Phase 34 non è stato confermato dalla scelta del team
sul feedback AIcrowd. Non ricaviamo il suo punteggio test da questi valori.

Il fit finale Phase 14 usa tutte le **328.135 righe etichettate** e produce
**109.379 predizioni: 14.746 positive e 94.633 negative**. Il CSV ha header
`Id,Prediction`, etichette −1/+1 e ID nell'ordine del sample ufficiale.
Il [record finale](../results/eda/final_phase14_submission.json) conserva
calibrazione, sorgente storico, conteggi e fingerprint della submission.
`python run.py` usa esattamente questo sorgente e questa soglia; la
[suite di validazione](../configs/experiments/final_phase14_validation.json)
seleziona esplicitamente lo stesso codice storico. Il
[riepilogo precedente](../results/eda/final_training_summary.json) descrive
il run di settembre con soglia 0.21086909970855153 e 14.728 positivi.

AIcrowd ha restituito **F1 0.452 e accuracy 0.878** per la nuova submission
Phase 14, secondo il feedback del team. La CV development è stata riusata
per molte scelte e il 20% iniziale è stato consultato dalle baseline.
Inoltre il confronto fra le due submission ha contribuito alla scelta
finale: il punteggio leaderboard segue questa selezione. La scelta interna
della soglia riduce una fonte di ottimismo, senza eliminare quello della
selezione ripetuta dei modelli.
