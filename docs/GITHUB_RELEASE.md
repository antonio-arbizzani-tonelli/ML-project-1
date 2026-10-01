# Preparazione della versione GitHub

La versione selezionata è Phase 14: 295 feature, `HAREHAB1` presente,
peso positivo 1 e soglia 0,21086909970855153. README e
[report finale](FINAL_REPORT.md) presentano il progetto e i risultati;
[EXPERIMENTS.md](EXPERIMENTS.md) raccoglie i confronti conclusi.

## Contenuti del commit

La versione comprende codice, test, configurazioni, codebook testuale,
documentazione, rapporti compatti e split development fisso. Dati ufficiali,
cache, checkpoint, OOF individuali, log e submission restano locali e ignorati.
[results/README.md](../results/README.md) descrive le evidenze distribuite e
gli artefatti necessari per riprodurre i singoli studi.

I 106 record JSON indicizzati conservano i byte originali: gli indici ne
riportano gli hash e `.gitattributes` ne mantiene gli LF. I percorsi assoluti
e gli hash dei sorgenti nei record descrivono l'esecuzione storica.
La riproduzione da un altro checkout usa le configurazioni CV e genera
nuovi artefatti locali. I runner amministrativi possono richiedere
checkpoint, OOF o copie storiche dei sorgenti, come indicato nella
[guida alle utility](../tools/README.md) e nei rapporti dello studio.

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

`tools.verify_repository` controlla tutti i file tracciati e i file non ignorati
destinati al commit, inclusi link locali, configurazione finale e hash dei
registri. La suite usa fixture e il codebook incluso, quindi può essere
eseguita senza scaricare il dataset.

La [CI](../.github/workflows/tests.yml) esegue i controlli con Python 3.9,
NumPy 1.23.1 e le dipendenze di sviluppo. Il suo esito sarà disponibile su
GitHub Actions dopo il push. Le verifiche locali completate sono registrate
in [FINAL_AUDIT.md](FINAL_AUDIT.md).

## Commit e push

La preparazione dei file è completa; staging, commit e push sono passaggi separati.

```bash
git add .gitignore .gitattributes .github README.md requirements.txt requirements-dev.txt implementations.py run.py run_mlp.py data/README.md configs docs src tests tools results/README.md results/eda results/experiments
git diff --cached --stat
git diff --cached --check
git diff --cached
git status --short
git commit -m "Finalize Phase 14 model, experiment report and MLP comparison"
git push origin HEAD
```

La selezione include le estensioni e le prove Phase 20–32 presenti nel working
tree. Dopo il push, controllare GitHub Actions. Per la consegna universitaria
usare il link del commit effettivo:

```text
https://github.com/antonio-arbizzani-tonelli/ML-project-1/tree/COMMIT_HASH
```

I test pubblici del corso e l'invio della submission seguono le istruzioni
ufficiali della consegna.

## Verifica di un nuovo checkout

Installare Python 3.9 e le dipendenze seguendo il [README](../README.md), poi
eseguire i controlli sopra. Per riprodurre la CV, scaricare i quattro CSV e
avviare `python -m tools.prepare_data` prima del relativo runner.
Per generare la submission selezionata usare `python run.py`.
Se cambia il contenuto dei CSV, usare una directory cache nuova.
