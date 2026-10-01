# MLP NumPy: preprocessing, training e valutazione

La MLP è una pipeline sperimentale separata. `run.py`, il preprocessing degli
alberi e `configs/final_model.json` continuano a definire il booster Phase 14.
La configurazione delle due reti è `configs/experiments/phase32_mlp.json`.

Il confronto Phase 32 è concluso. Entrambe le reti hanno un F1 inferiore a
Phase 14 in tutti e tre i fold e non sono state adottate:

| Pipeline | F1 annidato | Delta rispetto a Phase 14 | Fold favorevoli |
| --- | ---: | ---: | ---: |
| Phase 14, booster selezionato | 0.442155 | — | — |
| MLP 64→32 | 0.424451 | −0.017704 | 0/3 |
| MLP 128→64 | 0.419268 | −0.022887 | 0/3 |

Metriche, soglie e verifiche sono nel [rapporto Phase 32](../results/eda/phase32_mlp_summary.md)
e nel [confronto JSON](../results/experiments/mlp/phase32_comparison.json).
Il preprocessing e la configurazione finale del booster sono invariati.

## Codice

- `src/numpy_mlp.py`: ReLU, uscita sigmoide, BCE stabile sui logits,
  backpropagation, Adam, L2 sui soli pesi, mini-batch ed early stopping.
- `src/mlp_preprocessing.py`: rappresentazione dedicata e fitting sul training.
- `src/run_mlp_cv.py`: CV annidata, checkpoint con preprocessing, inferenza
  riproducibile, OOF interni/esterni e registro degli esperimenti.
- `run_mlp.py`: training su tutte le etichette e CSV con predizioni -1/+1,
  richiedendo una soglia già scelta sul development.

## Rappresentazione

La pulizia condivide il codebook e gli override Phase 14: stessi 295 predittori
ammessi, più il flag ausiliario di età mancante configurato nel piano. I valori
grezzi restano invariati. Il preprocessing degli alberi non viene modificato.

Continue e conteggi usano la mediana del training; ordinali e binarie usano la
moda. Numeriche e ordinali sono centrate e divise per la deviazione standard
calcolata dopo imputazione sul training. Binarie, indicatori e one-hot restano
0/1. Le nominali hanno il one-hot completo, un flag missing e un flag per
categorie non presenti nel training. Ogni numerica/binaria ha un flag missing.
I codici binari documentati sono conservati anche quando il training osserva
un solo livello. Colonne completamente mancanti usano un placeholder zero e
un flag; non si leggono statistiche della validazione. Non si aggiunge una
colonna di intercetta e non si applica il binning degli alberi.

Le nove frequenze codificate vengono convertite in occasioni/giorni per
giorno: `ALCDAY5`, `EXEROFT1/2` e sei colonne alimentari. Le risposte settimanali
usano il divisore 7, quelle mensili 30. Flag separati conservano l'unità della
risposta. Il codice alimentare 300, meno di una volta al mese, usa zero nel
canale numerico **e un flag distinto**; non gli viene attribuita una frequenza
esatta e resta distinguibile da Never. La semantica è verificata nel codebook
locale, pagine 32, 34–36, 40 e 43. Codici finiti inattesi causano un errore.

La larghezza è appresa per fold, perché dipende dalle categorie osservate.
Il preflight sul training esterno 1 produce 1.178 input: circa 77.569 parametri
per 64→32 e 159.233 per 128→64. Canali sempre zero nel training hanno pesi
iniziali zero, per evitare effetti casuali quando si attiva una categoria nuova.

## Protocollo

Stessi 262.508 esempi development, tre fold esterni (seed 20260919) e due
interni per fold esterno (seed 20260922 + numero esterno). Ogni soglia F1 usa
soltanto gli OOF interni, senza fissare 0,5 o trasferire la soglia del booster.

Ogni training viene ulteriormente diviso 90/10, stratificato, per scegliere
le epoche. Anche il preprocessing di questa selezione usa soltanto il 90%.
Si monitora la BCE non pesata, con massimo 80 epoche, patience 8 e min_delta
0,0001. Si conserva l'epoca con la BCE minima. Poi preprocessing e rete sono
rifatti da zero su **tutto il training di quel fit**, per quel numero di epoche.
I dati valutati esternamente non entrano mai in questa procedura.

Ogni architettura usa nove selezioni e nove refit completi, quindi il confronto
di due reti esegue 36 training. Adam usa learning rate 0,001, batch 512 e
seed 20261001. L'obiettivo è BCE media + 0,5 × 0,0001 × somma dei pesi al
quadrato, senza penalizzare i bias. Non si usano pesi di classe o resampling.

## Riproduzione

Installare le dipendenze del [README](../README.md) e collocare i quattro CSV
ufficiali come descritto in [data/README.md](../data/README.md). La CV legge
cache NumPy già preparate; non importa direttamente i CSV. Eseguire dalla
radice del progetto, in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m tools.prepare_data
python -m unittest discover -s tests -q
python -u -m src.run_mlp_cv configs/experiments/phase32_mlp.json
```

`tools.prepare_data` controlla l'allineamento degli ID e crea le cache sotto
`data/processed/` senza allenare modelli. Se cambiano i valori dei CSV delle
feature, preparare una nuova cartella di cache: il riuso controlla dimensioni
e ID, non ogni valore. La lista delle righe development è inclusa nel repository.

I fit producono curve di loss, confini esatti tra gradient fit/early stopping/
evaluation, audit delle feature e checkpoint sotto `results/mlp_artifacts/phase32/`.
Sono salvati sorgenti eseguiti, hash dei dati/configurazioni, OOF interni ed
esterni. I record sono nel registro separato `results/experiments/mlp/`.
Il tempo registrato per candidato è il tempo dell'intera suite condivisa;
i tempi dei singoli training sono disponibili nei report di fit.
Cache, checkpoint, OOF e report dei fit sono artefatti locali esclusi da Git;
i rapporti compatti e i record degli esperimenti restano consultabili nel repository.
I record storici conservano anche percorsi dell'ambiente di esecuzione:
il replay dei checkpoint richiede gli artefatti locali e percorsi coerenti.

La suite è stata eseguita in Python 3.14.3 / NumPy 2.4.2, con un thread
OpenBLAS. La compatibilità nell'ambiente ufficiale Python 3.9 / NumPy 1.23.1
resta da verificare. Il tempo complessivo registrato è 531,47 secondi;
dipende dall'hardware e dall'ambiente numerico.

`run_mlp.py` permette un training sperimentale completo dopo aver fissato
modello e soglia sul development. Per consultare gli argomenti:

```powershell
python run_mlp.py --help
```

`--model` accetta `mlp64_32` e `mlp128_64`; `--threshold` è obbligatorio e
richiede un numero tra 0 e 1. Nessuna soglia finale della MLP è stata adottata.
Il comando legge i CSV ufficiali, salva il preprocessing con la rete sotto
`results/mlp_artifacts/full_training/` e scrive per default `results/submission_mlp.csv`.
I checkpoint pickle devono essere caricati solo da sorgenti locali fidate.
L'inferenza da raw matrix è disponibile tramite `load_pipeline` e
`predict_pipeline` in `src/run_mlp_cv.py`.

## Interpretazione

La metrica principale è l'F1 aggregato con soglie annidate. Si riportano anche
F1 per fold, precision, recall, AP, log loss e tempi. Le due architetture hanno
lo stesso preprocessing. Il confronto con il booster valuta pipeline complete,
non il solo effetto dell'architettura a parità di codifica degli input.

Le numerose prove precedenti sul development rendono possibile ottimismo da
selezione ripetuta: questa CV non è un test finale intatto. Il confronto usa
un solo seed di inizializzazione e non dimostra che ogni MLP sia inferiore.
`HAREHAB1` è conservata nella rappresentazione; il suo legame con il target
limita l'interpretazione del risultato, come descritto nel [report finale](FINAL_REPORT.md).
La CV salva predizioni di validazione e non genera una submission.
