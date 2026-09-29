# Piano di sviluppo per migliorare l'F1

**Progetto:** classificazione binaria di `_MICHD` con logica NumPy-only.
**Metrica principale:** F1 della classe positiva.
**Ambiente ufficiale:** Python 3.9 e NumPy 1.23.1.
**Stato:** verifica Phase 14 e Phase 19 completate. I due trasferimenti T1–T2 peggiorano l'F1; il riferimento resta Phase 14.

Questo documento è la base operativa per i prossimi esperimenti. Se un risultato è già disponibile, va riutilizzato dopo averne verificato configurazione e protocollo. Ogni nuova prova deve lasciare un record con ipotesi, controllo, metriche, costo e decisione.

## 1. Riferimento e prove già svolte

Il riferimento è il booster a istogrammi NumPy della **Phase 14**, valutato sui 262.508 esempi development con tre fold esterni e due fold interni per scegliere la soglia F1.

| Voce | Valore |
| --- | ---: |
| Feature conservate | 295 |
| F1 annidato aggregato | **0,44216** |
| Precision | 0,36763 |
| Recall | 0,55457 |
| Alberi | 200 |
| Profondità massima | 5 |
| Learning rate | 0,05 |
| Minimo di righe per foglia | 200 |
| Bin | 64 |
| Regolarizzazione L2 delle foglie | 1 |
| Feature candidate per nodo | 128 |

La soglia di submission `0,2108690997` deriva da questo modello. **Ogni nuovo candidato deve selezionare la propria soglia** nei fold interni.

Artefatti principali: [configurazione finale](../configs/final_model.json), [protocollo e decisioni](EXPERIMENTS.md), [riferimento Phase 14](../configs/experiments/phase14_boosting_codebook_corrections_only.json).

### Evitare ripetizioni

- La Phase 11 ha completato 12 combinazioni di profondità 3/5, 100/200/400 alberi e 64/128 feature candidate, tutte con soglia annidata. La migliore profondità 3 raggiunge F1 `0,44099`; la migliore profondità 5 raggiunge `0,44174` prima delle correzioni Phase 14. Non ripetere quella griglia.
- Le Phase 3, 6 e 9 hanno esplorato indicatori di mancanza, one-hot, termini non lineari e interazioni **sulla logistica**. La Phase 3 era uno screening con 50 aggiornamenti e soglia fissa 0,5; questi risultati non chiudono le corrispondenti ipotesi sul booster.
- La Phase 13 ha rimosso più feature insieme e ottenuto F1 annidato `0,44029`. Non isola l'effetto di `ALCDAY5`.
- La Phase 16 ha riaddestrato varianti con imputazione ristretta ed estesa. I loro F1 disponibili scelgono la soglia sulle stesse predizioni OOF valutate e sono quindi **esplorativi**, non direttamente confrontabili con lo `0,44216` annidato. Si veda il [confronto Phase 16](../results/experiments/phase16_retrained_comparison.json).
- La [Phase 15](../configs/experiments/phase15_boosting_ablate_harehab1.json) ha già completato l'ablazione di `HAREHAB1`: F1 annidato `0,43104`, contro `0,44216` di Phase 14.
- La Phase 16 remota contiene cinque interazioni mirate nella configurazione, ma il preprocessing ad alberi versionato non legge il campo `interactions`: conserva le 295 colonne sorgente di quel ramo e il record dichiara 295 colonne in uscita. Il risultato non documenta una prova effettiva delle cinque interazioni. La Phase 17 ha invece confrontato realmente learning rate `0,03/0,05/0,07` e minimo foglia `150/200/250` su quel ramo senza `HAREHAB1`, con altre differenze di preprocessing rispetto alla Phase 15.
- La [Phase 18](../configs/experiments/phase18_boosting_diet_frequency_features.json) ha sostituito sei codici grezzi delle frequenze alimentari con i derivati giornalieri già presenti: F1 annidato `0,44186`, contro `0,44216` di Phase 14. La sostituzione completa non migliora il riferimento.
- La [Phase 19](../results/eda/phase19_transfer_summary.md) ha trasferito i due segnali Phase 17 alla Phase 14 con `HAREHAB1`, cambiando un solo parametro alla volta. Minimo foglia 150: F1 annidato `0,43999`; learning rate 0,07: `0,44070`. Entrambi peggiorano tutti e tre i fold rispetto a Phase 14 (`0,44216`). Non ripetere queste due configurazioni sullo stesso ramo.
- L'analisi degli errori condivisi fra logistica e boosting esiste già. Recuperarla prima di progettare un ensemble.

## 2. Prima fase: correttezza e preprocessing

### C0 — Consolidare la baseline

Prima dei nuovi training, verificare che configurazione, codice, split e artefatti Phase 14 descrivano lo stesso modello. Riutilizzare le predizioni salvate solo quando coincidono righe, ordine, preprocessing e parametri.

Verifica completata il 29 settembre 2026: tutti i 35 controlli sono passati, compresi gli hash degli input, gli split, il percorso effettivo del preprocessing e le predizioni dei tre checkpoint ricalcolate su tutte le righe di validazione. La differenza massima dalle predizioni salvate è zero in ogni fold. Il controllo mirato dei codici `777` delle dieci frequenze/durate alimentari e di esercizio conferma la conversione a `NaN`. Nessun modello è stato addestrato nella verifica. Il [rapporto di conformità](../results/eda/phase19_phase14_conformity.json) riporta anche l'ambiente effettivo: Python 3.14.3 e NumPy 2.4.2; l'ambiente ufficiale Python 3.9 / NumPy 1.23.1 resta un controllo distinto.

Rivedere le variabili di alimentazione e attività fisica per individuare **eventuali** codici speciali non ancora corretti dalle 14 conversioni Phase 14. Codici come `555` o `888` possono indicare «mai» in specifiche domande: il loro significato va verificato **per variabile** nel codebook. Distinguere zero reale, non risposta e domanda non applicabile; non applicare conversioni globali per valore numerico.

Una correzione semantica verificata resta necessaria anche se riduce l'F1. Se cambia gli input, misurarne l'effetto e fissare una nuova baseline corretta prima dei confronti successivi. Se non emerge alcun errore residuo, conservare Phase 14 senza riaddestrarla inutilmente.

### V1 — Ablazione separata di `HAREHAB1` (completata)

La Phase 15 ha riaddestrato il riferimento eliminando **solo** `HAREHAB1`: F1 annidato `0,43104`, precision `0,35002`, recall `0,56087`. Rispetto a Phase 14, l'F1 diminuisce di circa `0,01112`. La domanda è legata alla storia di infarto e potrebbe rivelare indirettamente il target tramite la logica del questionario. Il risultato quantifica la dipendenza dalla feature; non stabilisce da solo se sia ammissibile.

Verificarne l'ammissibilità rispetto al task e alle regole del progetto. Se va esclusa, la baseline senza `HAREHAB1` diventa il riferimento per tutti gli esperimenti successivi. Registrare i risultati con e senza la feature come due scenari espliciti; non ripetere l'ablazione già completata.

### Primo blocco di preprocessing

Mantenere fissi gli iperparametri del booster di riferimento. Valutare separatamente ogni modifica.

| ID | Modifica | Controllo | Domanda e stato iniziale |
| --- | --- | --- | --- |
| **P1** | Rimuovere soltanto `ALCDAY5`, conservando `DROCDY3_` | Baseline corretta | Il codice grezzo, che mescola frequenze settimanali e mensili, aggiunge informazione alla frequenza standardizzata? Da eseguire. |
| **P2** | One-hot mirato di `EXRACT11` e `EXRACT21`, sostituendo le colonne numeriche | Baseline corretta | L'ordine numerico arbitrario delle attività limita il booster? Da eseguire. |
| **P3** | Aggiungere gli indicatori di mancata risposta **identici** a quelli della Phase 16 ristretta | Baseline corretta | L'informazione sul motivo del mancante migliora l'F1? Esiste una configurazione Phase 15 di soli indicatori, ma va allineata a questo controllo. |
| **P4** | Aggiungere l'imputazione Phase 16 ristretta mantenendo gli stessi indicatori di P3 | P3 e baseline corretta | I valori ricostruiti aiutano oltre agli indicatori? Riaddestramento esplorativo già svolto; manca conferma annidata. |

**Dettagli di P1.** La prova rimuove una sola colonna e conserva tutte le altre feature Phase 14. La rimozione multipla Phase 13 non risponde a questa domanda. Non ricreare una conversione di `ALCDAY5` già rappresentata da `DROCDY3_`.

**Dettagli di P2.** Apprendere le categorie solo sul training del fold. Distinguere categorie sconosciute e mancanti. Conservare `EXRACT21=88` come «nessuna seconda attività», dove così definito, senza trattarlo come non risposta. Mantenere inizialmente 128 feature candidate per nodo: l'espansione one-hot cambia anche il campionamento delle colonne, quindi il risultato misura l'intera pipeline.

**Dettagli di P3 e P4.** Confrontare tre bracci sugli stessi split: baseline, baseline con indicatori, indicatori con imputazione ristretta. La configurazione Phase 15 di soli indicatori non è automaticamente equivalente agli indicatori Phase 16: controllare esattamente feature, codici e nomi. Addestrare imputatori e modello di malattia dentro ogni partizione di training; non passare loro etichette o valori del fold di validazione.

Lo screening Phase 16 fornisce F1 `0,44208` per il booster originale, `0,44267` per l'imputazione ristretta e `0,44199` per l'estesa **con soglia ottimizzata sulle stesse predizioni OOF**. Il vantaggio ristretto è piccolo e va confermato. La variante estesa non ha priorità.

### Secondo blocco di preprocessing

Procedere dopo il primo blocco, isolando ogni famiglia di modifiche.

| ID | Modifica | Condizione e controllo |
| --- | --- | --- |
| **P5** | Uniformare le unità delle frequenze alimentari | La sostituzione delle sei variabili grezze con i derivati giornalieri è **già stata valutata** in Phase 18: F1 `0,44186`, sotto Phase 14. Considerare altre varianti soltanto con una nuova ipotesi specifica. |
| **P6** | Uniformare le unità delle frequenze di esercizio | Valutare separatamente da P5 e mantenere distinti frequenza, tipo e assenza di attività. |
| **P7** | Conteggi o proporzioni di mancanti per modulo | Solo se i moduli sono identificabili nella documentazione. Verificare che la mancata somministrazione delle domande non ricostruisca direttamente il target. |

Preferire un derivato già disponibile quando rappresenta la stessa quantità. Non sostituire tutte le risposte alimentari con un unico totale.

### Trasformazioni a bassa priorità

Standardizzazione, logaritmi generici, clipping automatico e imputazione indiscriminata dei mancanti richiedono una motivazione legata a variabili precise. Il booster usa bin a quantili e gestisce i `NaN`; codici, unità e categorie offrono ipotesi più concrete da verificare prima.

## 3. Seconda fase: tuning del booster

La Phase 17 ha già eseguito cinque confronti **sul ramo Phase 16 senza `HAREHAB1`**. Il controllo di quel ramo ha F1 annidato `0,43137`; `learning_rate=0,03` dà `0,43085`, `0,07` dà `0,43222`, minimo foglia `150` dà **`0,43261`**, e minimo foglia `250` dà `0,43122`. Il titolo del commit privilegia il learning rate `0,07`, ma il minimo foglia `150` ha l'F1 più alto fra quelle prove. Nessuno di questi numeri va confrontato direttamente con Phase 14 come se le feature fossero identiche.

Fissare la migliore rappresentazione confermata e confrontare inizialmente le configurazioni seguenti **una per volta** sullo stesso ramo. Se si mantiene la baseline Phase 14, i primi due candidati trasferiscono su quel ramo i segnali della Phase 17. Se `HAREHAB1` va esclusa, i risultati Phase 17 sono riutilizzabili solo adottando esattamente la rappresentazione Phase 16; con la rappresentazione Phase 15 o una nuova variante, ripetere i confronti sul controllo corrispondente.

Il trasferimento dei primi due candidati è ora **completato** in Phase 19: T1 perde `0,002165` F1 e T2 perde `0,001455`, con zero fold favorevoli su tre per entrambi. Conservare minimo foglia 200 e learning rate 0,05 sul ramo Phase 14. Tornare al preprocessing P1–P4 prima di altri tuning; una nuova rappresentazione richiede un controllo corrispondente.

| ID | Modifica rispetto ai parametri Phase 14 | Ipotesi |
| --- | --- | --- |
| **T1** | `min_samples_leaf=150` | Completato su Phase 14: F1 `0,43999`; mantenere 200 sul ramo attuale. |
| **T2** | `learning_rate=0.07` | Completato su Phase 14: F1 `0,44070`; mantenere 0,05 sul ramo attuale. |
| **T3** | `learning_rate=0.025`, `n_estimators=400` | Distribuire l'apprendimento su aggiornamenti più piccoli. |
| **T4** | `l2_regularization=5` | Regolarizzare maggiormente le stime delle foglie. |
| **T5** | `max_depth=7` | Catturare interazioni non accessibili a profondità 5. |

T3 modifica intenzionalmente due parametri e **non ripete** la prova già svolta con 400 alberi e learning rate 0,05. Per T1, T2, T4 e T5, mantenere gli altri parametri del controllo scelto. Valori più estremi della dimensione delle foglie, come 100 o 400, passano a una fase successiva solo se i confronti vicino a 150–200 lo giustificano.

Non aprire una griglia cartesiana. Confermare i due candidati migliori; poi verificare **una sola combinazione** delle modifiche favorevoli. Un beneficio isolato non dimostra che la combinazione migliori l'F1.

## 4. Terza fase: metodi successivi

Avviare questa fase dopo preprocessing e tuning. Limitare il primo confronto ai tre candidati seguenti.

| ID | Metodo | Confronto e motivo |
| --- | --- | --- |
| **M1** | Peso dei positivi pari a 2 durante l'addestramento | Confrontare con peso 1. Applicare la pesatura in obiettivo, gradienti e Hessiani e riselezionare la soglia; un semplice spostamento dei punteggi non basta. |
| **M2** | Media uniforme di due booster con seed `20260920` e `20260921` | Verificare se il campionamento delle feature produce errori complementari. Registrare anche il costo aggiuntivo. |
| **M3** | Ensemble fra booster e logistica | Confrontare pesi della logistica 0,1, 0,2 e 0,3, selezionando peso e soglia solo nei fold interni. Allineare prima la logistica alla politica di imputazione corrente. |

L'analisi degli errori già salvata mostra molta sovrapposizione fra booster e logistica, perciò M3 viene dopo M1 e M2. I vecchi risultati lineari usavano mediane anche per alcune categorie: non attribuire all'ensemble un cambiamento causato dalla diversa imputazione.

Solo se gli esperimenti precedenti indicano un limite concreto della rappresentazione, valutare nell'ordine:

1. **Split categorici nativi** nel booster, se P2 è utile ma il one-hot è costoso.
2. **Target encoding con cross-fitting**, se categorie numerose restano difficili da rappresentare. Le etichette della riga valutata e del fold di validazione non devono contribuire alla sua codifica.
3. **128 bin** invece di 64, se l'analisi delle continue suggerisce una risoluzione insufficiente.

Random Forest, Extra Trees, focal loss e surrogati diretti dell'F1 non rientrano nel primo programma: richiedono sviluppo aggiuntivo senza un segnale specifico che siano il limite attuale.

## 5. Protocollo e criteri di decisione

### Validazione comune

- Usare gli stessi 262.508 esempi development e gli split già fissati: tre fold esterni, due fold interni per scegliere la soglia.
- Rifare statistiche, categorie, trasformazioni supervisionate, imputatori e modello **dentro il training** di ciascun livello.
- Nello screening usare inizialmente lo stesso primo fold esterno, mantenendo i due fold interni. Uno screening non è un nuovo risultato finale.
- Confermare sul totale dei tre fold i due candidati di preprocessing più promettenti. Se uno è P4, includere anche P3: massimo tre candidati di preprocessing da confermare.
- Per il tuning eseguire cinque screening e completare i tre fold per i due migliori. Valutare poi una combinazione completa.
- Limitare la prima fase dei nuovi metodi a M1–M3.
- Riutilizzare un fold o un artefatto salvato solo quando coincidono codice, configurazione, righe, ordine, preprocessing e confini di training.

La **metrica principale** è l'F1 aggregato sulle predizioni dei fold esterni, come nel riferimento Phase 14. Riportare separatamente F1 per fold, media dei fold, precision, recall, AP, log loss, soglie e tempi.

### Regole operative

- Promuovere un candidato se l'F1 aggregato migliora e il segno del miglioramento è favorevole in almeno due fold su tre.
- Registrare come incerto un guadagno concentrato in un solo fold. La regola dei due fold è un filtro operativo, non una prova statistica.
- Considerare `+0,002` F1 assoluto un segnale interessante, non un requisito rigido: un guadagno minore, coerente e poco costoso può essere mantenuto.
- Per attribuire un beneficio all'imputazione, confrontare P4 direttamente con P3, oltre che con la baseline.
- Valutare esplicitamente ogni combinazione o ensemble. Se nessun candidato supera il controllo, chiudere la fase conservando il riferimento.
- Tenere distinto il risultato dell'ablazione `HAREHAB1` dai candidati selezionati per F1.

Il 20% iniziale è già stato consultato dalle baseline, secondo [EXPERIMENTS.md](EXPERIMENTS.md). Non presentarlo come holdout intatto. Anche la selezione ripetuta sui fold development può rendere ottimistiche le stime finali.

### Controlli prima di un training completo

Verificare su casi mirati: codici speciali e zeri reali, colonne tutte mancanti, categorie sconosciute, isolamento dei fold, allineamento di righe ed etichette, nomi e ordine delle feature, ricerca della soglia in presenza di punteggi identici, riproducibilità e compatibilità con NumPy 1.23.1.

La CV completa del booster attuale richiede circa **46,6 minuti** sulla macchina di riferimento. Registrare il tempo effettivo di ogni prova; 400 alberi, profondità 7 ed ensemble possono costare sensibilmente di più.

## 6. Registro operativo

Aggiornare questa tabella quando una prova termina. I risultati numerici vanno collegati al relativo record JSON; non sostituire il protocollo annidato con un F1 esplorativo nella colonna decisione.

| ID | Stato iniziale | Prossima azione | Risultato e decisione |
| --- | --- | --- | --- |
| C0 | Verifica di conformità completata | Mantenere il riferimento verificato per i confronti isolati | 35/35 controlli; predizioni identiche in tutti i fold; controllo mirato dei codici superato. |
| V1 | Completato | Decidere l'ammissibilità della feature, senza ripetere il training | Phase 15: F1 annidato `0,43104` senza `HAREHAB1`; Phase 14: `0,44216` con la feature. |
| P1 | Da eseguire | Ablazione isolata `ALCDAY5` | — |
| P2 | Da eseguire | One-hot delle due attività | — |
| P3 | Predisposto in parte | Allineare e valutare i soli indicatori Phase 16 ristretta | — |
| P4 | Screening esplorativo svolto | Conferma annidata con controllo P3 | — |
| P5 | Completato per sei frequenze alimentari | Conservare come riferimento; nuove varianti richiedono ipotesi precise | Phase 18: F1 annidato `0,44186`, sotto Phase 14. |
| P6–P7 | Da eseguire dopo P1–P4 | Frequenze di esercizio e mancanti per modulo | — |
| T1–T2 | Completati in Phase 19 sul ramo Phase 14 con `HAREHAB1` | Conservare i parametri Phase 14; procedere al preprocessing P1–P4 | T1: F1 `0,43999` (delta `−0,002165`); T2: `0,44070` (delta `−0,001455`); entrambi inferiori nei tre fold. |
| T3–T5 | Da eseguire dopo preprocessing | Tuning contenuto del booster | — |
| M1–M3 | Fase successiva | Pesi e ensemble | — |

Per ogni record annotare: data, ipotesi, configurazione del controllo, unica modifica introdotta, split e seed, numero di feature, F1 aggregato e per fold, precision, recall, AP, log loss, soglie, tempo, memoria, percorsi degli artefatti e decisione motivata.
