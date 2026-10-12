# Audit della versione finale — 12 ottobre 2026

La versione di consegna seleziona **Phase 14 storica**, con 295 feature,
200 alberi di profondità 5 e soglia **0.2106169795070252** calibrata su
OOF a tre fold di tutte le 328.135 righe train. Il team ha comunicato
**F1 0.452 e accuracy 0.878 su AIcrowd**. Il
[record finale](../results/eda/final_phase14_submission.json) conserva
provenienza del feedback, parametri, calibrazione e fingerprint della submission.
Il [report inglese](FINAL_REPORT.pdf) è di due pagine A4, senza appendice;
il [sorgente LaTeX](FINAL_REPORT.tex) è incluso.

## Modello, risultati e riproduzione

| Controllo | Esito |
| --- | --- |
| Selezione del codice | `run.py` e la suite finale CV usano `src/numpy_boosting_phase14.py`, copia byte-esatta del codice storico |
| Fingerprint del sorgente | `6fede57915b07a6ea7675292833f8ad79269e665f09977bffd2a9562ce1aaf98`; LF preservati da Git |
| F1 development annidato | 0.4421552272688187, con soglie scelte nei fold interni |
| Soglia di submission | 0.2106169795070252, da tre fold OOF su tutto il train; preprocessing appreso nel training di ciascun fold |
| F1 di calibrazione | 0.44276954874267993; le stesse etichette selezionano la soglia, quindi non è un test indipendente |
| Riproduzione del modello completo | Checkpoint finale caricato con il modulo normale Phase 14; matrici train/test ricostruite identiche, differenza massima delle probabilità 0 |
| Riproduzione del CSV | SHA-256 identico alla submission scelta: `8583daf81f7afb378eef3eafdfefd988ce85f78de8d58d38cd8ca39e1faa0df1` |
| Scrittura del CSV | `create_csv_submission` in `helpers.py`, adattata dallo starter EPFL con UTF-8 e LF espliciti; `run.py` conserva i controlli su forma e dominio delle etichette |
| Conteggi e schema | 109.379 ID unici nell'ordine del sample ufficiale; 14.746 positivi, 94.633 negativi; header `Id,Prediction`, etichette −1/+1 |
| Dati di input | Nove fingerprint dei CSV e delle cache verificati prima del replay |
| Entry point di training | Test end-to-end su fixture del `run_pipeline` normale con Phase 14 e creazione di checkpoint e submission |

Il training completo delle due implementazioni era già terminato: sei fit
OOF e due fit su tutte le righe, con identici parametri e fold. Il replay
verifica questi modelli e la pipeline finale senza ripetere i fit completi.
La nuova suite `configs/experiments/final_phase14_validation.json` rende
riproducibile la CV storica senza dipendere da sorgenti locali ignorati.

Phase 34 conserva la correzione degli split osservati/mancanti come confronto.
Il feedback leaderboard ha contribuito alla scelta finale fra le due
submission; le metriche development, calibrazione e AIcrowd sono separate
nel report. Il punteggio numerico AIcrowd di Phase 34 non è stato fornito.

## Verifica del contenuto per GitHub

- **206 test del progetto superati**, sia nel working tree sia in una copia
  del solo contenuto Git eleggibile, priva di dati, cache e checkpoint.
- **26 test pubblici del corso superati** contro la copia di consegna.
  Esclusi il controllo dell'URL, perché il test usa un percorso locale, e
  Black, che il test ufficiale dichiara facoltativo. La verifica dell'URL
  richiede il commit effettivamente pubblicato.
- Le sei funzioni di `implementations.py` superano i controlli ufficiali su
  firme, docstring, loss e aggiornamenti; GD/SGD, least squares, ridge e
  logistica sono implementati in NumPy.
- `tools.verify_repository` verifica sintassi Python 3.9, JSON, link Markdown
  locali, configurazione finale, codice congelato e tutti i **114 hash**
  dei record nei cinque registri.
- I quattro help (`run.py`, `run_mlp.py`, `tools.prepare_data`,
  `tools.final_phase14_phase34`) funzionano anche nella copia pulita.
- `git diff --check` passa. Modelli e preprocessing usano NumPy e libreria
  standard; Matplotlib è usato per analisi e grafici.
- Dati, cache, checkpoint, OOF individuali, log e submission sono esclusi dal
  contenuto Git eleggibile. Il pacchetto è di circa 21,5 MiB.
- Nessuna corrispondenza nella scansione dei pattern di chiavi private,
  token GitHub/OpenAI e access key AWS; il perimetro è indicato nel
  [riepilogo automatico](../results/eda/final_phase14_readiness.json).
- Il PDF compila senza errori né avvisi di testo fuori margine; entrambe le
  pagine renderizzate sono state controllate visivamente.

Ambiente delle esecuzioni locali: **Python 3.14.3 / NumPy 2.4.2**.
Il parsing della sintassi Python 3.9 non sostituisce un'esecuzione completa
nel runtime ufficiale. La CI è configurata per Python 3.9 / NumPy 1.23.1.

## Passaggi esterni per completare la consegna

Il contenuto locale è pronto per staging, commit e push; questi passaggi
non sono stati eseguiti durante l'audit. Dopo il push occorre:

1. controllare l'esito della CI nell'ambiente ufficiale;
2. verificare che il repository sia pubblico e accessibile al correttore;
3. consegnare il link al commit preciso e il PDF secondo le istruzioni del corso.

La visibilità remota non è stata verificata: l'accesso alla pagina/API GitHub
non era disponibile in questa sessione. La
[guida alla pubblicazione](GITHUB_RELEASE.md) contiene i comandi e il formato
richiesto del link. Il feedback AIcrowd è stato registrato dal messaggio del
team, senza URL della submission.

Gli [audit precedenti](../results/eda/final_readiness.json) e il
[controllo Phase 34](../results/eda/phase34_repository_verification.json)
restano evidenze datate; non definiscono il modello finale corrente.
