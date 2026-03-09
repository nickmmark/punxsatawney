# punxsatawney


## Doing the analysis
### Three ways to run the analysis
#### Uses embedded dataset (no files needed)
```
python punxsutawney_phil_analysis.py
```

#### Load data from the CSV
```
python punxsutawney_phil_analysis.py --csv phil_predictions_1900_2025.csv
```

#### Custom output dir and rolling window
```
python punxsutawney_phil_analysis.py --csv data.csv --outdir ./results --window 20
```

### Outputs
* summary_table.csv — all metrics per definition (accuracy, AUC, sensitivity, specificity, PPV, NPV, F1, MCC, p-values)
* roc_plot.png — multi-definition ROC operating points
* confusion_matrices.png — 4-panel confusion matrices
* auc_accuracy_bars.png — bar comparison with significance stars
* rolling_accuracy.png — N-year rolling accuracy per definition
* decadal_heatmap.png — decade-by-decade performance grid
* bias_chart.png — Phil's shadow rate vs. actual winter rate over time
* yearly_strip.png — full year-by-year outcome visualization
