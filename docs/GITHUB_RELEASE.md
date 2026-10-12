# Preparazione della versione GitHub

La versione finale è **Phase 14 storica**, con 295 feature, `HAREHAB1`
presente, 200 alberi, profondità 5 e peso positivo 1. La soglia
**0.2106169795070252** è calibrata su OOF a tre fold di tutte le 328.135
righe train. Il team ha comunicato **F1 0.452 e accuracy 0.878 su AIcrowd**.
Il [report PDF](FINAL_REPORT.pdf) contiene due pagine; le
[note estese](FINAL_REPORT.md) e [EXPERIMENTS.md](EXPERIMENTS.md) conservano
protocolli e confronti.

## Contenuti della versione

Il repository comprende codice, test, configurazioni, codebook testuale,
report PDF e LaTeX, rapporti compatti e split development fisso. Il sorgente
storico `src/numpy_boosting_phase14.py` è incluso e ha il fingerprint della
submission selezionata; `.gitattributes` ne preserva gli LF.
`run.py` e `configs/experiments/final_phase14_validation.json` lo selezionano
esplicitamente. Phase 34 resta disponibile come confronto corretto.
La scrittura del CSV passa da `create_csv_submission` in `helpers.py`,
adattata dallo starter del corso con UTF-8 e LF espliciti per mantenere
identico il file scelto.

Dati ufficiali, cache, checkpoint, OOF individuali, log e submission CSV
sono ignorati da Git. Il [record finale](../results/eda/final_phase14_submission.json)
conserva invece soglia, score AIcrowd comunicati, conteggi e hash del CSV.
I **114 record JSON in cinque registri** mantengono i byte originali e
gli hash verificati. I percorsi e gli hash nei record storici descrivono
l'esecuzione originale; i runner di studio possono richiedere artefatti
locali come indicato nella [guida alle utility](../tools/README.md).

## Verifica prima del commit

Dalla radice, con le dipendenze di sviluppo installate:

```bash
python -m tools.verify_repository
python -m unittest discover -s tests -q
python run.py --help
python run_mlp.py --help
python -m tools.prepare_data --help
git diff --check
```

I controlli usano fixture e il codebook incluso, quindi funzionano senza
scaricare il dataset. L'[audit finale](FINAL_AUDIT.md) registra anche la
verifica su una copia del solo contenuto Git eleggibile, i test pubblici del
corso e la riproduzione del CSV scelto. I test pubblici sono stati eseguiti
localmente: il controllo dell'URL del commit va completato sul link effettivo;
il suggerimento Black dei test ufficiali è esplicitamente facoltativo.

La [CI](../.github/workflows/tests.yml) usa Python 3.9 e NumPy 1.23.1.
Il suo esito va controllato dopo il push; i test locali usano Python 3.14.3
e NumPy 2.4.2. La sintassi Python 3.9 è controllata su tutti i sorgenti.

## Commit, push e link di consegna

Staging, commit e push non sono stati eseguiti durante questo audit.
Questi comandi includono tutti i file non ignorati verificati:

```bash
git add .
git diff --cached --stat
git diff --cached --check
git diff --cached
git status --short
git commit -m "Finalize Phase 14 submission and AIcrowd report"
git push origin HEAD
git rev-parse HEAD
```

Dopo il push, controllare GitHub Actions e rendere il repository pubblico
come richiesto dalla consegna. Usare il commit effettivo nel link:

```text
https://github.com/antonio-arbizzani-tonelli/ML-project-1/tree/COMMIT_HASH
```

Il report PDF è la consegna scritta; le note Markdown restano documentazione
del repository. La submission AIcrowd è caricata dal team.

## Riproduzione da un nuovo checkout

Seguire il [README](../README.md) per ambiente e download dei quattro CSV.
`python -m tools.prepare_data` prepara le cache; la suite
`configs/experiments/final_phase14_validation.json` riproduce la valutazione
development storica. `python run.py` addestra su tutto il train e genera
la submission Phase 14 con la soglia finale già calibrata. Se cambia il
contenuto dei CSV, usare una directory cache nuova. La calibrazione finale
di entrambe le versioni è riproducibile con `tools.final_phase14_phase34`.
