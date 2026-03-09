# punxsatawney

## Methods
I explored whether Punxsutawney Phil's annual Groundhog Day predictions -shadow (more winter) or no shadow (early spring) — contain statistically significant meteorological signal when evaluated against empirical Pennsylvania climate data.

### Predictions
I analyzed all of Phil's predictions from 1900 to present (n=124 because Phil did not make a prediction in 1943 or 1943 "War clouds have blacked out parts of the shadow.") Primary source: [groundhog.org](https://www.groundhog.org/groundhog-day/history-past-predictions/)
I used several definitions of "Early spring" including:

### Endpoints
All four endpoints use Pennsylvania statewide data to maintain geographic relevance to Punxsutawney (Punxsutawney County, western PA). Each is encoded as 1 = more winter / cold, 0 = early spring / warm.

#### NOAA CPC Statewide Seasonal Classification
Source: NOAA NCEI Climate at a Glance — Pennsylvania Statewide Temperature
The combined February–March mean temperature for Pennsylvania was ranked against the 1901–2000 baseline (long-record normal, preferred for pre-modern era coverage). Years classified as "Above Normal" (top tercile) = 0 (early spring); "Below Normal" or "Near Normal" = 1 (more winter). This mirrors the methodology used by NOAA's own Grading the Groundhogs analysis.

#### Temperature Anomaly vs. 30-Year Normal
Source: NOAA NCEI Climate at a Glance — Pennsylvania Statewide Temperature; NOAA U.S. Climate Normals 1991–2020
February 2–March 15 mean temperature compared to the 1901–2000 climatological mean (~37.5°F for western PA). A positive anomaly (warmer than normal) = 0; zero or negative anomaly = 1. This is a continuous rather than tercile-based measure, making it more sensitive to marginal years.

#### Heating Degree Days (HDD) Accumulation
Source: NOAA NCEI Climate at a Glance — Heating Degree Days, Pennsylvania; NOAA Climate Normals
Accumulated HDD (base 65°F) for the February 2–March 15 window compared to the climatological long-period mean for Pennsylvania. HDD above the historical mean = 1 (more heating demand = colder = more winter); below = 0. HDD is a standard energy and climate metric that integrates the full temperature distribution rather than relying on the mean alone, and thus captures cold extremes more sensitively than mean anomaly.

#### First Sustained Warm Day (≥50°F for 3+ Consecutive Days)
Source: NOAA GHCN-Daily station data — western Pennsylvania composite; primary station USW00094823 (Johnstown/Cambria County area) supplemented with nearby stations for gap-filling.

This definition operationalizes "early spring" as the arrival of the first sustained warm spell in the post-Groundhog Day window. A run of ≥3 consecutive days with maximum temperature ≥50°F occurring before March 1 = 0 (early spring); first such run occurring on or after March 1, or no qualifying run before April 1, = 1 (more winter). This threshold was chosen to reflect the folk-meteorological intuition underlying the tradition — a brief warm streak that would colloquially signal spring's arrival. Of the four definitions, this one most directly operationalizes what Phil's shadow forecast is predicting.

#### Composite (Majority Vote)
A derived variable: coded 1 if ≥3 of the four definitions independently classify the year as "more winter," 0 otherwise. This provides a consensus ground truth less sensitive to any single definition's classification errors.


## Results

| Definition           | N   | Acc   | AUC   | Sens  | Spec  | PPV   | F1    | MCC   | p(binom) |     |
|----------------------|-----|-------|-------|-------|-------|-------|-------|-------|----------|-----|
| CPC Classification   | 124 | 67.7% | 0.623 | 95.8% | 28.8% | 65.1% | 0.775 | 0.346 | 0.00005  | *** |
| Temp Anomaly         | 124 | 69.4% | 0.609 | 93.6% | 28.3% | 68.9% | 0.793 | 0.300 | <0.00001 | *** |
| Heating Degree Days  | 124 | 72.6% | 0.624 | 93.9% | 31.0% | 72.6% | 0.819 | 0.334 | <0.00001 | *** |
| First Warm Day ≥50°F | 124 | 84.7% | 0.710 | 93.9% | 48.0% | 87.7% | 0.907 | 0.478 | <0.00001 | *** |
| Composite            | 124 | 73.4% | 0.628 | 94.0% | 31.7% | 73.6% | 0.825 | 0.343 | <0.00001 | *** |

## Running the analysis
### Three ways to run the analysis
#### Use embedded dataset (no files needed)
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
