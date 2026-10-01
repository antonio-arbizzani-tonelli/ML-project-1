# Audit della versione finale

Verifica del 1 ottobre 2026 sui file tracciati e non ignorati destinati a
GitHub. Il modello selezionato è Phase 14, con 295 feature e `HAREHAB1`
presente. I risultati dei confronti sono nel [report finale](FINAL_REPORT.md).

## Coerenza del modello e dei risultati

| Controllo | Esito |
| --- | --- |
| Configurazione finale | Parametri uguali alla CV Phase 14; soglia 0,21086909970855153, media delle tre soglie interne |
| F1 development ricalcolato | 0,4421552272688187; TP 12.855, FP 22.112, TN 217.216, FN 10.325 |
| Replay Phase 14 | Tre checkpoint con preprocessing e sorgenti attuali: differenza massima delle probabilità 0 |
| Replay MLP | Tutti i 18 checkpoint interni/esterni: differenza massima delle probabilità 0; soglie e metriche ricostruite |
| Submission locale | 109.379 ID unici nell'ordine ufficiale; 14.728 positivi e 94.651 negativi |

I replay sono verifiche dei modelli salvati durante il consolidamento del
progetto. Il [riepilogo del training completo](../results/eda/final_training_summary.json)
conserva durata, conteggi e fingerprint della submission:
`f5fe9fe80b4db737bde4e690ad8ba296155dbf991ff136431c068d8fd77b32d4`.

## Verifica del pacchetto GitHub

- **164 test superati**, anche in una copia del solo contenuto destinato a
  GitHub, senza dati, cache o checkpoint preesistenti.
- Configurazioni, link Markdown locali, parsing JSON e tutti i **106 hash**
  dei record indicizzati verificati da `tools.verify_repository`.
- Sintassi Python 3.9 verificata per tutti i sorgenti. I modelli usano NumPy
  e libreria standard; Matplotlib è una dipendenza separata per EDA/test.
- `tools.prepare_data` crea le cinque cache da piccoli CSV e rifiuta un
  sample con ID in ordine errato. Sui dati ufficiali ha verificato 328.135
  righe train, 109.379 test e 321 feature grezze.
- Help dei tre comandi principali e `git diff --check` superati.
- Dati, cache, checkpoint, OOF individuali e submission esclusi dai file
  Git eleggibili. Il pacchetto è di circa 20 MiB.

Il [riepilogo automatico](../results/eda/final_readiness.json) riporta perimetro,
ambiente, conteggi e controlli eseguiti. La scansione dei pattern di credenziali
indicati nel riepilogo non ha trovato corrispondenze.

README e guide di cartella identificano modello finale, configurazioni,
risultati e comandi. Gli studi successivi sono indicati come conclusi;
i record storici conservano metriche, hash e provenienza originali.

## Verifiche esterne ancora da eseguire

| Verifica | Stato |
| --- | --- |
| Runtime ufficiale | Test locali eseguiti con Python 3.14.3 / NumPy 2.4.2. La CI è configurata per Python 3.9 / NumPy 1.23.1; esito da osservare dopo il push |
| Test pubblici del corso | Esecuzione con i materiali ufficiali ancora da registrare |
| Score della submission | Punteggio AIcrowd ancora da registrare |
