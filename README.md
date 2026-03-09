# punxsatawney


## Doing the analysis
### Uses embedded dataset (no files needed)
```bash
python punxsutawney_phil_analysis.py
```

### Load data from the CSV
```bash
python punxsutawney_phil_analysis.py --csv phil_predictions_1900_2025.csv
```

### Custom output dir and rolling window
```bash
python punxsutawney_phil_analysis.py --csv data.csv --outdir ./results --window 20
```
