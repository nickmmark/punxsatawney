# punxsatawney


## Results

| Definition           | N   | Acc   | AUC   | Sens  | Spec  | PPV   | F1    | MCC   | p(binom) |     |
|----------------------|-----|-------|-------|-------|-------|-------|-------|-------|----------|-----|
| CPC Classification   | 124 | 67.7% | 0.623 | 95.8% | 28.8% | 65.1% | 0.775 | 0.346 | 0.00005  | *** |
| Temp Anomaly         | 124 | 69.4% | 0.609 | 93.6% | 28.3% | 68.9% | 0.793 | 0.300 | <0.00001 | *** |
| Heating Degree Days  | 124 | 72.6% | 0.624 | 93.9% | 31.0% | 72.6% | 0.819 | 0.334 | <0.00001 | *** |
| First Warm Day ≥50°F | 124 | 84.7% | 0.710 | 93.9% | 48.0% | 87.7% | 0.907 | 0.478 | <0.00001 | *** |
| Composite            | 124 | 73.4% | 0.628 | 94.0% | 31.7% | 73.6% | 0.825 | 0.343 | <0.00001 | *** |

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
