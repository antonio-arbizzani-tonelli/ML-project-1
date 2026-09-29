# Audit completo dell’imputazione delle variabili

## Come leggere le metriche

Questa tabella riporta le medie semplici dei tre fold. Per variabili categoriche, Score = accuratezza esatta del modello / accuratezza della moda; per variabili continue, Score = MAE del modello / MAE della mediana. Un Δ positivo è favorevole per accuratezza, balanced accuracy e macro-F1; per MAE è favorevole un Δ negativo. “Non so/rifiuto” e “blank” sono conteggi nei tre fold di validazione sommati: i blank comprendono mancate somministrazioni o ramificazioni del questionario, che non vanno imputate come rifiuti. La valutazione usa risposte note oscurate artificialmente: non misura direttamente la correttezza sui veri non rispondenti.

Per lo score categorico mostro la media delle accuratezze dei fold; il riepilogo precedente usava l’accuratezza aggregata sulle righe, da cui piccole differenze possibili (61 contro 62 variabili sopra la moda). Balanced accuracy e macro-F1 sono medie dei tre fold.

## Proposta per il modello dell’infarto

Non abbiamo ancora misurato se queste imputazioni migliorano F1 o AP della malattia. Le decisioni qui definiscono i challenger da confrontare, non una modifica già dimostrata del modello finale.

- **Set principale:** imputare `INCOME2`, `PNEUVAC3`, `EMPLOY1`, `HTM4` e `WTKG3`; conservare i flag originali. Per `_BMI5`, usare la formula BRFSS da altezza e peso quando entrambi sono disponibili, invece di predirlo indipendentemente. Così evitiamo BMI incompatibili con altezza/peso.

- **Set esteso, da provare separatamente:** `ARTHDIS2`, `BPHIGH4`, `EDUCA`, `FLUSHOT6`, `GENHLTH`, `HAVARTH3`, `HPVTEST`, `JOINPAIN`, `LMTJOIN3`, `MARITAL`, `PDIABTST`, `QLACTLM2`, `RCSRLTN2`, `RENTHOM1`, `SMOKE100`, `TOLDHI2`. Queste 16 variabili — incluse `BPHIGH4` e `GENHLTH`, con forti guadagni ma meno di 1.000 non rispondenti — migliorano tutte e tre le metriche rispetto alla moda in ciascun fold; serve comunque verificare l’effetto sul target infarto.

- **Non sostituire con una classe dura:** `HIVTST6`. L’accuratezza è leggermente peggiore della moda (0,699 contro 0,703), anche se balanced accuracy e macro-F1 migliorano. Per ora terrei `NaN` e il flag; si può confrontare separatamente un input probabilistico.

- **Lasciare mancanti:** tutte le altre variabili categoriche e `_FRUTSUM`, `_VEGESUM`. Altezza, peso e BMI hanno MAE nettamente migliore della mediana; il vantaggio della verdura è trascurabile e per la frutta la mediana è migliore. Per tutte, i blank strutturali restano mancanti.

- **Variabili aggiunte solo nel set derived/administrative:** non includerle per ora. Se un campo è calcolato da risposte sorgente, preferire la formula documentata; un campo amministrativo va valutato solo dopo aver confermato che sarà disponibile al momento della predizione.

## Set diretto completo — 171 categoriche e 5 continue

| Variabile | Origine | Tipo | Non so/rifiuto (dev) | Blank (dev) | Risposte note validate | Score modello / baseline | Δ score | Δ balanced acc. | Δ macro-F1 | Fold migliori | Decisione |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `ADANXEV` | optional_module | binary | 80 | 250,483 | 11,945 | 0.818 / 0.839 | -0.021 | 0.115 | 0.170 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADDEPEV2` | core_survey | binary | 1,156 | 0 | 261,352 | 0.799 / 0.811 | -0.012 | 0.120 | 0.184 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADDOWN` | optional_module | count | 141 | 250,430 | 11,937 | 0.712 / 0.743 | -0.030 | 0.027 | 0.022 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADEAT1` | optional_module | count | 201 | 250,444 | 11,863 | 0.624 / 0.650 | -0.026 | 0.020 | 0.021 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADENERGY` | optional_module | count | 228 | 250,438 | 11,842 | 0.429 / 0.394 | 0.035 | 0.029 | 0.036 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADFAIL` | optional_module | count | 140 | 250,448 | 11,920 | 0.796 / 0.830 | -0.034 | 0.026 | 0.021 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADMOVE` | optional_module | count | 247 | 250,468 | 11,793 | 0.851 / 0.889 | -0.038 | 0.022 | 0.017 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADPLEASR` | optional_module | count | 414 | 250,417 | 11,677 | 0.671 / 0.697 | -0.026 | 0.029 | 0.027 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADSLEEP` | optional_module | count | 178 | 250,435 | 11,895 | 0.540 / 0.539 | 0.001 | 0.025 | 0.034 | acc 2/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ADTHINK` | optional_module | count | 124 | 250,459 | 11,925 | 0.817 / 0.852 | -0.034 | 0.022 | 0.017 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ALCDAY5` | core_survey | count | 3,382 | 9,491 | 249,635 | 0.458 / 0.495 | -0.037 | 0.011 | 0.014 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ARTHDIS2` | core_survey | binary | 2,351 | 181,607 | 78,550 | 0.729 / 0.681 | 0.048 | 0.168 | 0.269 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `ARTHEDU` | optional_module | binary | 59 | 250,850 | 11,599 | 0.869 / 0.870 | -0.000 | 0.000 | 0.001 | acc 0/3; bal 1/3; macro 1/3 | Lasciare NaN nel modello ad albero |
| `ARTHEXER` | optional_module | binary | 151 | 250,843 | 11,514 | 0.599 / 0.578 | 0.021 | 0.080 | 0.213 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ARTHSOCL` | core_survey | ordinal | 688 | 181,733 | 80,087 | 0.621 / 0.576 | 0.044 | 0.188 | 0.256 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ARTHWGT` | optional_module | binary | 86 | 250,840 | 11,582 | 0.742 / 0.637 | 0.105 | 0.229 | 0.336 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ARTTODAY` | optional_module | categorical | 81 | 250,831 | 11,596 | 0.483 / 0.433 | 0.050 | 0.197 | 0.296 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASACTLIM` | optional_module | ordinal | 22 | 262,103 | 383 | 0.736 / 0.743 | -0.008 | 0.019 | 0.019 | acc 0/3; bal 1/3; macro 1/3 | Lasciare NaN nel modello ad albero |
| `ASDRVIST` | optional_module | count | 5 | 262,318 | 185 | 0.491 / 0.578 | -0.087 | -0.009 | 0.016 | acc 0/3; bal 1/3; macro 2/3 | Lasciare NaN nel modello ad albero |
| `ASERVIST` | optional_module | count | 4 | 262,318 | 186 | 0.633 / 0.683 | -0.050 | 0.017 | 0.045 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASINHALR` | optional_module | ordinal | 7 | 262,105 | 396 | 0.457 / 0.475 | -0.019 | 0.018 | 0.056 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASNOSLEP` | optional_module | ordinal | 10 | 262,227 | 271 | 0.438 / 0.477 | -0.039 | 0.082 | 0.125 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASRCHKUP` | optional_module | count | 10 | 262,103 | 395 | 0.314 / 0.368 | -0.054 | 0.004 | 0.038 | acc 0/3; bal 2/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASTHMA3` | core_survey | categorical | 807 | 0 | 261,701 | 0.847 / 0.866 | -0.018 | 0.049 | 0.093 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASTHMAGE` | optional_module | count | 55 | 261,875 | 578 | 0.322 / 0.366 | -0.044 | 0.004 | 0.005 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASTHMED3` | optional_module | count | 12 | 262,105 | 391 | 0.473 / 0.348 | 0.125 | 0.100 | 0.209 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASTHNOW` | core_survey | binary | 1,068 | 227,327 | 34,113 | 0.658 / 0.692 | -0.034 | 0.095 | 0.187 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `ASYMPTOM` | optional_module | ordinal | 10 | 262,104 | 394 | 0.266 / 0.312 | -0.046 | 0.029 | 0.113 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `AVEDRNK2` | core_survey | count | 2,092 | 136,701 | 123,715 | 0.466 / 0.470 | -0.004 | 0.012 | 0.022 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `BLDSUGAR` | optional_module | ordinal | 305 | 245,106 | 17,097 | 0.244 / 0.260 | -0.016 | 0.005 | 0.012 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `BLIND` | core_survey | binary | 974 | 6,666 | 254,868 | 0.898 / 0.951 | -0.053 | 0.150 | 0.117 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `BLOODCHO` | core_survey | categorical | 5,592 | 0 | 256,916 | 0.876 / 0.886 | -0.010 | 0.208 | 0.232 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `BPHIGH4` | core_survey | categorical | 757 | 0 | 261,751 | 0.706 / 0.580 | 0.126 | 0.113 | 0.181 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `BPMEDS` | core_survey | binary | 169 | 157,027 | 105,312 | 0.831 / 0.837 | -0.006 | 0.155 | 0.211 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CASTHDX2` | optional_module | categorical | 1,490 | 231,327 | 29,691 | 0.851 / 0.870 | -0.019 | 0.021 | 0.052 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CASTHNO2` | optional_module | binary | 96 | 258,659 | 3,753 | 0.650 / 0.673 | -0.023 | 0.011 | 0.073 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CDDISCUS` | optional_module | binary | 74 | 254,822 | 7,612 | 0.630 / 0.554 | 0.076 | 0.118 | 0.261 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CDHELP` | optional_module | ordinal | 20 | 260,056 | 2,432 | 0.389 / 0.415 | -0.026 | 0.043 | 0.114 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CDHOUSE` | optional_module | ordinal | 187 | 254,790 | 7,531 | 0.520 / 0.489 | 0.030 | 0.081 | 0.122 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CDSOCIAL` | optional_module | ordinal | 144 | 254,814 | 7,550 | 0.512 / 0.519 | -0.006 | 0.102 | 0.132 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHCCOPD1` | core_survey | binary | 1,240 | 0 | 261,268 | 0.870 / 0.920 | -0.051 | 0.188 | 0.167 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHCKIDNY` | core_survey | binary | 790 | 0 | 261,718 | 0.921 / 0.965 | -0.044 | 0.105 | 0.083 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHCOCNCR` | core_survey | binary | 546 | 0 | 261,962 | 0.839 / 0.901 | -0.062 | 0.074 | 0.096 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHCSCNCR` | core_survey | binary | 658 | 1 | 261,849 | 0.864 / 0.905 | -0.041 | 0.078 | 0.107 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHECKUP1` | core_survey | ordinal | 3,567 | 1 | 258,940 | 0.715 / 0.747 | -0.033 | 0.042 | 0.074 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHILDREN` | core_survey | count | 2,188 | 3 | 260,317 | 0.710 / 0.740 | -0.029 | 0.037 | 0.040 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHKHEMO3` | optional_module | ordinal | 1,154 | 245,108 | 16,246 | 0.287 / 0.290 | -0.003 | 0.017 | 0.028 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CHOLCHK` | core_survey | ordinal | 3,448 | 34,996 | 224,064 | 0.736 / 0.779 | -0.042 | 0.038 | 0.063 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CIMEMLOS` | optional_module | categorical | 485 | 193,126 | 68,897 | 0.852 / 0.893 | -0.041 | 0.100 | 0.114 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CPDEMO1` | core_survey | binary | 767 | 111,642 | 150,099 | 0.773 / 0.813 | -0.040 | 0.162 | 0.202 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CRGVEXPT` | optional_module | binary | 3,704 | 212,514 | 46,290 | 0.834 / 0.834 | 0.000 | 0.000 | 0.000 | acc 0/3; bal 0/3; macro 0/3 | Lasciare NaN nel modello ad albero |
| `CRGVHOUS` | optional_module | binary | 78 | 248,295 | 14,135 | 0.772 / 0.779 | -0.006 | 0.022 | 0.058 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CRGVHRS1` | optional_module | ordinal | 840 | 248,249 | 13,419 | 0.548 / 0.578 | -0.030 | 0.039 | 0.072 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CRGVLNG1` | optional_module | ordinal | 193 | 248,229 | 14,086 | 0.281 / 0.297 | -0.017 | 0.020 | 0.092 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CRGVMST2` | optional_module | categorical | 444 | 248,334 | 13,730 | 0.828 / 0.828 | -0.001 | 0.000 | 0.001 | acc 1/3; bal 2/3; macro 2/3 | Lasciare NaN nel modello ad albero |
| `CRGVPERS` | optional_module | binary | 81 | 248,287 | 14,140 | 0.563 / 0.507 | 0.056 | 0.062 | 0.226 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CRGVPRB1` | optional_module | categorical | 674 | 248,266 | 13,568 | 0.458 / 0.470 | -0.013 | 0.002 | 0.006 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CRGVREL1` | optional_module | categorical | 98 | 248,216 | 14,194 | 0.305 / 0.224 | 0.081 | 0.133 | 0.135 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CVDASPRN` | optional_module | binary | 17 | 251,205 | 11,286 | 0.718 / 0.706 | 0.013 | 0.183 | 0.261 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `CVDSTRK3` | core_survey | binary | 571 | 0 | 261,937 | 0.900 / 0.960 | -0.059 | 0.137 | 0.094 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DECIDE` | core_survey | binary | 1,660 | 7,022 | 253,826 | 0.872 / 0.903 | -0.031 | 0.180 | 0.190 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIABAGE2` | core_survey | count | 2,209 | 228,639 | 31,660 | 0.096 / 0.091 | 0.005 | 0.019 | 0.025 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIABEDU` | optional_module | binary | 74 | 245,109 | 17,325 | 0.595 / 0.560 | 0.035 | 0.079 | 0.218 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIABETE3` | core_survey | categorical | 363 | 4 | 262,141 | 0.807 / 0.845 | -0.038 | 0.101 | 0.110 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIABEYE` | optional_module | binary | 210 | 245,109 | 17,189 | 0.803 / 0.821 | -0.018 | 0.028 | 0.070 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIFFALON` | core_survey | binary | 1,117 | 7,954 | 253,437 | 0.888 / 0.922 | -0.035 | 0.247 | 0.211 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIFFDRES` | core_survey | binary | 678 | 7,590 | 254,240 | 0.915 / 0.957 | -0.041 | 0.237 | 0.167 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DIFFWALK` | core_survey | binary | 1,304 | 7,364 | 253,840 | 0.844 / 0.827 | 0.017 | 0.276 | 0.298 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DOCTDIAB` | optional_module | count | 614 | 245,108 | 16,786 | 0.255 / 0.260 | -0.004 | 0.009 | 0.019 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DRADVISE` | optional_module | binary | 151 | 238,058 | 24,299 | 0.720 / 0.698 | 0.022 | 0.177 | 0.263 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `DRNK3GE5` | core_survey | count | 1,744 | 136,940 | 123,824 | 0.728 / 0.757 | -0.028 | 0.005 | 0.007 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EDUCA` | core_survey | ordinal | 1,034 | 0 | 261,474 | 0.452 / 0.369 | 0.083 | 0.116 | 0.184 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `EMPLOY1` | core_survey | categorical | 2,250 | 0 | 260,258 | 0.611 / 0.412 | 0.199 | 0.245 | 0.260 | acc 3/3; bal 3/3; macro 3/3 | Set principale: imputare nel prossimo challenger |
| `EMTSUPRT` | optional_module | ordinal | 197 | 250,475 | 11,836 | 0.510 / 0.544 | -0.034 | 0.067 | 0.113 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXERANY2` | core_survey | binary | 1,481 | 21,097 | 239,930 | 0.713 / 0.734 | -0.021 | 0.120 | 0.199 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXERHMM1` | core_survey | count | 3,540 | 87,835 | 171,133 | 0.245 / 0.253 | -0.008 | 0.002 | 0.004 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXERHMM2` | core_survey | count | 2,625 | 145,298 | 114,585 | 0.218 / 0.223 | -0.005 | 0.001 | 0.004 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXEROFT1` | core_survey | count | 1,701 | 87,622 | 173,185 | 0.184 / 0.180 | 0.004 | 0.003 | 0.005 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXEROFT2` | core_survey | count | 1,325 | 145,189 | 115,994 | 0.174 / 0.177 | -0.003 | 0.003 | 0.006 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXRACT11` | core_survey | categorical | 710 | 86,575 | 175,223 | 0.514 / 0.548 | -0.035 | 0.011 | 0.010 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EXRACT21` | core_survey | categorical | 2,223 | 88,065 | 172,220 | 0.257 / 0.318 | -0.061 | 0.013 | 0.014 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `EYEEXAM` | optional_module | categorical | 272 | 245,108 | 17,128 | 0.489 / 0.511 | -0.022 | 0.022 | 0.049 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FEETCHK` | optional_module | count | 507 | 245,272 | 16,729 | 0.243 / 0.226 | 0.017 | 0.007 | 0.024 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FEETCHK2` | optional_module | ordinal | 513 | 245,108 | 16,887 | 0.521 / 0.532 | -0.012 | 0.002 | 0.004 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FLUSHOT6` | core_survey | categorical | 1,795 | 24,612 | 236,101 | 0.621 / 0.519 | 0.103 | 0.119 | 0.278 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `FRUIT1` | core_survey | count | 4,262 | 17,366 | 240,880 | 0.234 / 0.251 | -0.017 | 0.003 | 0.005 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FRUITJU1` | core_survey | count | 6,152 | 16,700 | 239,656 | 0.405 / 0.412 | -0.007 | 0.001 | 0.003 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FVBEANS` | core_survey | count | 5,511 | 18,089 | 238,908 | 0.168 / 0.171 | -0.003 | 0.004 | 0.007 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FVGREEN` | core_survey | count | 3,811 | 18,626 | 240,071 | 0.147 / 0.141 | 0.006 | 0.005 | 0.007 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `FVORANG` | core_survey | count | 4,166 | 19,119 | 239,223 | 0.164 / 0.148 | 0.016 | 0.004 | 0.009 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `GENHLTH` | core_survey | ordinal | 709 | 2 | 261,797 | 0.399 / 0.331 | 0.069 | 0.172 | 0.275 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `HADHYST2` | optional_module | binary | 69 | 249,085 | 13,354 | 0.711 / 0.719 | -0.009 | 0.164 | 0.238 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HADSGCO1` | optional_module | binary | 470 | 239,138 | 22,900 | 0.959 / 0.959 | -0.000 | -0.000 | -0.000 | acc 0/3; bal 0/3; macro 0/3 | Lasciare NaN nel modello ad albero |
| `HAREHAB1` | optional_module | binary | 13 | 261,888 | 607 | 0.589 / 0.604 | -0.015 | 0.069 | 0.193 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HAVARTH3` | core_survey | binary | 1,500 | 1 | 261,007 | 0.722 / 0.664 | 0.058 | 0.198 | 0.295 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `HIVTST6` | core_survey | categorical | 8,540 | 25,801 | 228,167 | 0.699 / 0.703 | -0.004 | 0.160 | 0.240 | acc 0/3; bal 3/3; macro 3/3 | Non sostituire con classe dura |
| `HLTHPLN1` | core_survey | binary | 1,039 | 0 | 261,469 | 0.905 / 0.927 | -0.022 | 0.121 | 0.147 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HOWLONG` | optional_module | ordinal | 103 | 251,442 | 10,963 | 0.604 / 0.622 | -0.018 | 0.022 | 0.041 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HPLSTTST` | optional_module | ordinal | 224 | 258,392 | 3,892 | 0.453 / 0.463 | -0.010 | 0.049 | 0.086 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HPVADSHT` | optional_module | count | 139 | 261,635 | 734 | 0.594 / 0.630 | -0.037 | 0.034 | 0.093 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HPVADVC2` | optional_module | categorical | 623 | 256,118 | 5,767 | 0.836 / 0.847 | -0.011 | 0.152 | 0.184 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `HPVTEST` | optional_module | count | 3,894 | 248,961 | 9,653 | 0.676 / 0.573 | 0.103 | 0.181 | 0.310 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `HTM4` | derived | continuous | 0 | 9,096 | 253,412 | 0.033 / 0.086 | -0.054 | — | — | — | Set principale: imputare nel prossimo challenger |
| `IMFVPLAC` | core_survey | categorical | 389 | 148,959 | 113,160 | 0.407 / 0.381 | 0.027 | 0.090 | 0.092 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `INCOME2` | core_survey | ordinal | 45,073 | 1,947 | 215,488 | 0.372 / 0.320 | 0.052 | 0.138 | 0.184 | acc 3/3; bal 3/3; macro 3/3 | Set principale: imputare nel prossimo challenger |
| `INSULIN` | optional_module | binary | 19 | 245,105 | 17,384 | 0.658 / 0.671 | -0.013 | 0.046 | 0.134 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `INTERNET` | core_survey | binary | 826 | 2,561 | 259,121 | 0.819 / 0.791 | 0.028 | 0.265 | 0.303 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `JOINPAIN` | core_survey | count | 1,571 | 183,244 | 77,693 | 0.194 / 0.161 | 0.033 | 0.074 | 0.122 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `LASTPAP2` | optional_module | ordinal | 313 | 249,855 | 12,340 | 0.464 / 0.414 | 0.051 | 0.091 | 0.118 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `LASTSIG3` | optional_module | ordinal | 317 | 239,140 | 23,051 | 0.232 / 0.216 | 0.017 | 0.032 | 0.114 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `LASTSMK2` | core_survey | ordinal | 499 | 190,004 | 72,005 | 0.671 / 0.668 | 0.003 | 0.052 | 0.066 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `LENGEXAM` | optional_module | ordinal | 28 | 259,537 | 2,943 | 0.587 / 0.621 | -0.033 | 0.026 | 0.046 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `LMTJOIN3` | core_survey | binary | 1,010 | 181,504 | 79,994 | 0.655 / 0.505 | 0.150 | 0.154 | 0.318 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `LONGWTCH` | optional_module | count | 850 | 247,495 | 14,163 | 0.133 / 0.122 | 0.011 | 0.006 | 0.011 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `LSATISFY` | optional_module | ordinal | 133 | 250,484 | 11,891 | 0.587 / 0.487 | 0.100 | 0.137 | 0.226 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `LSTBLDS3` | optional_module | ordinal | 345 | 250,762 | 11,401 | 0.375 / 0.355 | 0.020 | 0.030 | 0.075 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `MARITAL` | core_survey | categorical | 1,801 | 0 | 260,707 | 0.595 / 0.533 | 0.062 | 0.196 | 0.241 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `MAXDRNKS` | core_survey | count | 3,854 | 137,191 | 121,463 | 0.303 / 0.284 | 0.019 | 0.014 | 0.024 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `MEDCOST` | core_survey | binary | 666 | 1 | 261,841 | 0.869 / 0.902 | -0.033 | 0.098 | 0.131 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `MENTHLTH` | core_survey | count | 4,299 | 0 | 258,209 | 0.676 / 0.695 | -0.019 | 0.012 | 0.010 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `MISTMNT` | optional_module | binary | 50 | 250,479 | 11,979 | 0.829 / 0.853 | -0.024 | 0.115 | 0.165 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `NUMHHOL2` | core_survey | categorical | 579 | 111,642 | 150,287 | 0.947 / 0.947 | 0.000 | 0.000 | 0.000 | acc 1/3; bal 2/3; macro 2/3 | Lasciare NaN nel modello ad albero |
| `NUMPHON2` | core_survey | count | 174 | 254,607 | 7,727 | 0.535 / 0.520 | 0.015 | 0.025 | 0.071 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PCPSAAD2` | optional_module | binary | 164 | 258,652 | 3,692 | 0.677 / 0.604 | 0.073 | 0.152 | 0.277 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PCPSADE1` | optional_module | categorical | 4 | 262,040 | 464 | 0.305 / 0.335 | -0.030 | 0.012 | 0.130 | acc 0/3; bal 2/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PCPSADI1` | optional_module | binary | 183 | 258,657 | 3,668 | 0.698 / 0.712 | -0.014 | 0.021 | 0.074 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PCPSARE1` | optional_module | binary | 153 | 258,660 | 3,695 | 0.667 / 0.540 | 0.127 | 0.162 | 0.312 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PCPSARS1` | optional_module | categorical | 17 | 260,431 | 2,060 | 0.724 / 0.749 | -0.025 | 0.009 | 0.025 | acc 0/3; bal 2/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PDIABTST` | optional_module | binary | 2,243 | 213,372 | 46,893 | 0.675 / 0.644 | 0.030 | 0.111 | 0.222 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `PERSDOC2` | core_survey | categorical | 1,060 | 0 | 261,448 | 0.757 / 0.777 | -0.020 | 0.104 | 0.130 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PHYSHLTH` | core_survey | count | 5,684 | 1 | 256,823 | 0.647 / 0.637 | 0.010 | 0.020 | 0.017 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PNEUVAC3` | core_survey | binary | 21,380 | 24,950 | 216,178 | 0.737 / 0.568 | 0.169 | 0.230 | 0.369 | acc 3/3; bal 3/3; macro 3/3 | Set principale: imputare nel prossimo challenger |
| `POORHLTH` | core_survey | count | 2,845 | 128,022 | 131,641 | 0.563 / 0.568 | -0.005 | 0.017 | 0.014 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PREDIAB1` | optional_module | categorical | 179 | 213,371 | 48,958 | 0.877 / 0.884 | -0.008 | 0.121 | 0.166 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PREGNANT` | core_survey | binary | 291 | 223,949 | 38,268 | 0.966 / 0.961 | 0.004 | 0.126 | 0.185 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `PSATIME` | optional_module | ordinal | 46 | 260,429 | 2,033 | 0.617 / 0.643 | -0.027 | 0.023 | 0.054 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `QLACTLM2` | core_survey | binary | 1,758 | 5,566 | 255,184 | 0.787 / 0.753 | 0.034 | 0.216 | 0.285 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `RCSGENDR` | optional_module | binary | 1,700 | 226,893 | 33,915 | 0.510 / 0.518 | -0.007 | 0.003 | 0.142 | acc 0/3; bal 2/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `RCSRLTN2` | optional_module | categorical | 1,354 | 227,025 | 34,129 | 0.847 / 0.815 | 0.032 | 0.256 | 0.240 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `RDUCHART` | optional_module | binary | 96 | 259,184 | 3,228 | 0.857 / 0.852 | 0.005 | 0.108 | 0.171 | acc 2/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `RDUCSTRK` | optional_module | binary | 244 | 259,185 | 3,079 | 0.723 / 0.720 | 0.003 | 0.064 | 0.139 | acc 2/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `RENTHOM1` | core_survey | categorical | 1,815 | 0 | 260,693 | 0.772 / 0.723 | 0.049 | 0.227 | 0.295 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `RLIVPAIN` | optional_module | binary | 18 | 259,185 | 3,305 | 0.783 / 0.790 | -0.008 | 0.077 | 0.145 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SCNTLPAD` | optional_module | categorical | 417 | 247,313 | 14,778 | 0.620 / 0.453 | 0.167 | 0.099 | 0.174 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SCNTLWK1` | optional_module | count | 1,669 | 239,786 | 21,053 | 0.389 / 0.439 | -0.050 | 0.008 | 0.011 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SCNTMEL1` | optional_module | ordinal | 148 | 218,871 | 43,489 | 0.650 / 0.668 | -0.018 | 0.104 | 0.120 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SCNTMNY1` | optional_module | ordinal | 157 | 221,141 | 41,210 | 0.527 / 0.545 | -0.017 | 0.118 | 0.143 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SCNTPAID` | optional_module | categorical | 254 | 241,701 | 20,553 | 0.631 / 0.473 | 0.158 | 0.226 | 0.328 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SEATBELT` | core_survey | ordinal | 1,385 | 24,065 | 237,058 | 0.863 / 0.875 | -0.012 | 0.015 | 0.017 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SHINGLE2` | optional_module | binary | 264 | 245,817 | 16,427 | 0.710 / 0.713 | -0.003 | 0.143 | 0.227 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SMOKDAY2` | core_survey | ordinal | 198 | 153,250 | 109,060 | 0.681 / 0.665 | 0.016 | 0.121 | 0.182 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SMOKE100` | core_survey | categorical | 1,944 | 8,496 | 252,068 | 0.622 / 0.566 | 0.055 | 0.110 | 0.249 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `STREHAB1` | optional_module | binary | 2 | 262,101 | 405 | 0.610 / 0.625 | -0.015 | 0.054 | 0.167 | acc 1/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `STRENGTH` | core_survey | count | 3,075 | 23,053 | 236,380 | 0.584 / 0.615 | -0.031 | 0.003 | 0.005 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `SXORIENT` | optional_module | categorical | 2,785 | 163,287 | 96,436 | 0.964 / 0.968 | -0.004 | 0.007 | 0.012 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `TETANUS` | optional_module | categorical | 8,347 | 238,357 | 15,804 | 0.622 / 0.649 | -0.027 | 0.089 | 0.138 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `TOLDHI2` | core_survey | binary | 2,062 | 34,996 | 225,450 | 0.664 / 0.579 | 0.084 | 0.156 | 0.289 | acc 3/3; bal 3/3; macro 3/3 | Set esteso: testare come secondo challenger |
| `TRNSGNDR` | optional_module | categorical | 1,435 | 163,341 | 97,732 | 0.995 / 0.995 | -0.000 | -0.000 | -0.000 | acc 0/3; bal 0/3; macro 0/3 | Lasciare NaN nel modello ad albero |
| `USEEQUIP` | core_survey | binary | 612 | 6,221 | 255,675 | 0.854 / 0.884 | -0.030 | 0.248 | 0.234 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `USENOW3` | core_survey | ordinal | 1,145 | 8,868 | 252,495 | 0.959 / 0.968 | -0.009 | 0.024 | 0.032 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VEGETAB1` | core_survey | count | 4,837 | 19,721 | 237,950 | 0.214 / 0.231 | -0.017 | 0.003 | 0.004 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VETERAN3` | core_survey | binary | 493 | 0 | 262,015 | 0.855 / 0.869 | -0.014 | 0.247 | 0.253 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VICTRCT4` | optional_module | categorical | 6 | 260,607 | 1,895 | 0.662 / 0.626 | 0.036 | 0.223 | 0.286 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIDFCLT2` | optional_module | ordinal | 1 | 260,595 | 1,912 | 0.720 / 0.781 | -0.060 | 0.018 | 0.037 | acc 0/3; bal 2/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIEYEXM2` | optional_module | categorical | 41 | 260,988 | 1,479 | 0.436 / 0.448 | -0.011 | 0.061 | 0.120 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIGLUMA2` | optional_module | binary | 9 | 260,607 | 1,892 | 0.901 / 0.932 | -0.032 | 0.027 | 0.049 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIINSUR2` | optional_module | categorical | 51 | 260,606 | 1,851 | 0.620 / 0.568 | 0.052 | 0.096 | 0.220 | acc 3/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIMACDG2` | optional_module | binary | 19 | 260,608 | 1,881 | 0.922 / 0.938 | -0.016 | 0.026 | 0.045 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VINOCRE2` | optional_module | categorical | 13 | 261,848 | 647 | 0.457 / 0.410 | 0.048 | 0.060 | 0.104 | acc 2/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIPRFVS2` | optional_module | categorical | 19 | 260,601 | 1,888 | 0.444 / 0.499 | -0.055 | 0.026 | 0.079 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `VIREDIF3` | optional_module | ordinal | 10 | 260,601 | 1,897 | 0.533 / 0.585 | -0.052 | 0.032 | 0.065 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `WHRTST10` | core_survey | categorical | 1,345 | 194,840 | 66,323 | 0.430 / 0.469 | -0.039 | 0.052 | 0.078 | acc 0/3; bal 3/3; macro 3/3 | Lasciare NaN nel modello ad albero |
| `WTKG3` | derived | continuous | 0 | 18,364 | 244,144 | 5.196 / 15.797 | -10.602 | — | — | — | Set principale: imputare nel prossimo challenger |
| `_BMI5` | derived | continuous | 0 | 21,655 | 240,853 | 1.492 / 4.637 | -3.145 | — | — | — | Derivare da HTM4 e WTKG3 |
| `_FRUTSUM` | derived | continuous | 0 | 25,821 | 236,687 | 0.826 / 0.801 | 0.025 | — | — | — | Lasciare NaN nel modello ad albero |
| `_VEGESUM` | derived | continuous | 0 | 30,202 | 232,306 | 0.914 / 0.923 | -0.009 | — | — | — | Lasciare NaN nel modello ad albero |

## Variabili aggiuntive del set derived/administrative esteso

Sono presenti solo nella variante estesa. I blank possono riflettere il flusso del questionario; le frequenze di attività perdono rispetto alla mediana.

| Variabile | Origine | Tipo | Non so/rifiuto (dev) | Blank (dev) | Risposte note validate | Score modello / baseline | Δ score | Δ balanced acc. | Δ macro-F1 | Fold migliori | Decisione |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `DRNKANY5` | derived | binary | 1,869 | 0 | 249,635 | 0.660 / 0.505 | 0.154 | 0.159 | 0.323 | acc 3/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `HHADULT` | administrative | count | 337 | 153,915 | 108,256 | 0.588 / 0.508 | 0.079 | 0.037 | 0.054 | acc 3/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `LANDLINE` | administrative | binary | 200 | 151,324 | 110,984 | 0.684 / 0.622 | 0.061 | 0.161 | 0.278 | acc 3/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `PAFREQ1_` | derived | continuous | 0 | 89,323 | 173,185 | 1.940 / 1.936 | 0.004 | — | — | — | Non includere nel set proposto |
| `PAFREQ2_` | derived | continuous | 0 | 146,514 | 115,994 | 1.869 / 1.803 | 0.066 | — | — | — | Non includere nel set proposto |
| `PAMISS1_` | derived | categorical | 22,578 | 0 | 227,174 | 1.000 / 1.000 | 0.000 | 0.000 | 0.000 | acc 0/3; bal 0/3; macro 0/3 | Non includere nel set proposto |
| `STRFREQ_` | derived | continuous | 0 | 26,128 | 236,380 | 1.410 / 1.201 | 0.209 | — | — | — | Non includere nel set proposto |
| `_AIDTST3` | derived | binary | 8,540 | 25,801 | 228,167 | 0.699 / 0.703 | -0.004 | 0.160 | 0.240 | acc 0/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `_MRACE1` | derived | categorical | 5,180 | 0 | 257,328 | 0.797 / 0.832 | -0.035 | 0.067 | 0.089 | acc 0/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `_PRACE1` | derived | categorical | 5,180 | 0 | 257,328 | 0.806 / 0.841 | -0.035 | 0.060 | 0.081 | acc 0/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `_RACE` | derived | categorical | 4,366 | 0 | 258,142 | 0.758 / 0.776 | -0.017 | 0.083 | 0.104 | acc 0/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `_RACEG21` | derived | binary | 4,366 | 0 | 258,142 | 0.785 / 0.776 | 0.009 | 0.148 | 0.223 | acc 3/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
| `_RACEGR3` | derived | categorical | 4,366 | 0 | 258,142 | 0.765 / 0.776 | -0.011 | 0.108 | 0.144 | acc 0/3; bal 3/3; macro 3/3 | Non includere nel set proposto |
