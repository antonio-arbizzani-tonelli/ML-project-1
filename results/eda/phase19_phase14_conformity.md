# Verifica di conformità della Phase 14

Verifica completata il 29 settembre 2026 senza addestrare modelli. Tutti i **35 controlli** sono passati. Il [JSON completo](phase19_phase14_conformity.json) conserva hash, risultati dei controlli e nomi delle feature per fold.

| Controllo | Esito |
| --- | --- |
| Dataset e cache | Hash identici al record Phase 14; 328.135 righe e 321 colonne grezze. |
| Development | 262.508 righe, 23.180 positivi; righe, ordine, etichette e assegnazione ai fold identici. |
| Configurazione finale | Parametri identici al modello Phase 14 salvato. |
| Preprocessing | Stesso percorso ad alberi, 295 feature, `HAREHAB1` conservata, imputazione assente. |
| Implementazione al 29 settembre 2026 | Booster, valutazione e runner erano invariati rispetto al commit iniziale; le aggiunte locali al preprocessing non erano attive nella Phase 14. |
| Checkpoint e predizioni OOF | Hash corretti. Le predizioni ricalcolate sui tre fold hanno differenza massima **0** rispetto alle predizioni salvate. |
| Soglia finale | `0.21086909970855153`, identica alla media delle tre soglie interne Phase 14. |
| Metriche annidate | F1, precision, recall e altre metriche di classificazione ricostruite dalle predizioni salvate coincidono con il record. |
| Codici mirati | `777` delle sei frequenze alimentari e delle quattro frequenze/durate di esercizio viene convertito a `NaN`, come previsto dal codebook. |

Seed conservati: `20260918` per lo split iniziale, `20260919` per i fold esterni, `20260920` per il booster e `20260922 + numero del fold esterno` per i fold interni.

L'ambiente locale effettivo è Python **3.14.3** con NumPy **2.4.2**. La riproduzione delle predizioni verifica questo ambiente; non certifica una nuova esecuzione nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1.

La verifica dimostra la corrispondenza con il riferimento salvato alla data del controllo. Le successive estensioni sperimentali dei sorgenti sono distinte da quella verifica di uguaglianza del codice. La [verifica finale](../../docs/FINAL_AUDIT.md) ha nuovamente riprodotto le probabilità dei tre checkpoint Phase 14 con i sorgenti aggiornati. `HAREHAB1` è conservata nella versione finale; frequenze e attività sono state esaminate nei confronti successivi descritti nel [report](../../docs/FINAL_REPORT.md).

I due confronti successivi, minimo foglia **150** e learning rate **0.07**, sono stati completati separatamente contro Phase 14, con soglie scelte nei fold interni. Gli [esiti Phase 19](phase19_transfer_summary.md) non ne hanno motivato l'adozione; la combinazione dei due parametri non è stata eseguita.
