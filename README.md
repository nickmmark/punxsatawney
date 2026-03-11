# punxsatawney

Every February 2nd, a groundhog known as Punxsutawney Phil emerges in Pennsylvania to render a meteorological verdict: if he sees his shadow, legend says winter will persist for six more weeks; if not, spring will arrive early. The ritual is charming folklore (and a good [movie](https://en.wikipedia.org/wiki/Groundhog_Day_(film))), but it also offers a surprisingly useful framework for thinking about prediction in medicine.

At its core, Phil’s prediction is a binary diagnostic test. The “test result” is whether Phil sees his shadow. The “outcome” is whether the following weeks resemble winter or spring. When framed this way, Phil can be evaluated with the same tools we use for clinical prediction models: sensitivity, specificity, PPV, calibration, and ROC curves. Does a shadow actually correlate with colder weather? How often does Phil correctly predict early spring? Is he better than chance—or better than a trivial baseline, like always predicting six more weeks of winter?

This exercise mirrors the challenges we face when evaluating medical predictors. In critical care we constantly rely on prognosticators: labs, imaging, risk scores, etc. A lactate level predicts mortality; a Wells score predicts pulmonary embolism; an echocardiographic finding like VTI predicts shock physiology. In each case, the real question is the same as Phil’s: how good is the predictor when tested _rigorously_?

Two lessons emerge. First, prediction must be benchmarked against a clear ground truth. For Phil, that requires defining “spring” meteorologically and rigorously. Medicine faces the same challenge when defining outcomes like sepsis, ARDS, or neurological recovery. The wrong definition can change the prediction.

Second, a predictor must outperform simpler alternatives. In many cases, naïve baselines perform surprisingly well. If winter conditions persist most years through mid-March, then _always_ predicting “six more weeks of winter” might achieve higher accuracy. Many medical prediction rules fail this same test: they sound sophisticated but add little information beyond basic clinical judgment or prevalence.

Finally, rigorous evaluation reminds us that prediction is probability not prophecy. Even excellent clinical models rarely achieve perfect discrimination. Phil’s forecasts, like many medical predictors, may ultimately perform little better than chance. But the value of evaluating them carefully is not to ridicule the predictor—it is to sharpen our understanding of uncertainty.

Punxsutawney Phil is less a meteorologist than a teaching tool. By subjecting his shadow-based forecast to the same statistical scrutiny we apply to clinical prediction models, we are reminded that the essential question in both weather and medicine is the same: does the signal truly predict the outcome, or are we simply seeing shadows?

![](https://upload.wikimedia.org/wikipedia/commons/6/6e/Punxsutawney_Phil_2018_%28cropped%29.jpg?_=20190711105238)


## Methods
I explored whether Punxsutawney Phil's annual Groundhog Day predictions - shadow (more winter) or no shadow (early spring) - contain statistically significant meteorological signal when evaluated against empirical Pennsylvania climate data.
I calculated the sensitivity, specificity, positive predictive value, F1 score, area under the curve for a received operating characteristic, and overall accuracy.

Significance testing was performed comparing Phil's predictions to chance (50% probability of getting it right; e.g. a coin flip), using a [Bernoulli trial methodology](https://en.wikipedia.org/wiki/Bernoulli_trial).

I also used chi-squared.



## Data Sources
### Phil's predictions
I analyzed all of Phil's predictions from 1900 to present (n=124 because Phil did not make a prediction in 1943 or 1943 "War clouds have blacked out parts of the shadow.") Primary source: [groundhog.org](https://www.groundhog.org/groundhog-day/history-past-predictions/)

I used several definitions of "Early spring" including:

### Meterological Endpoints
All four endpoints use Pennsylvania statewide data to maintain geographic relevance to Punxsutawney (Punxsutawney County, western PA). Each is encoded as 1 = more winter / cold, 0 = early spring / warm.

#### NOAA CPC Statewide Seasonal Classification
* The combined February–March mean temperature for Pennsylvania was ranked against the 1901–2000 baseline (long-record normal, preferred for pre-modern era coverage).
* Years classified as "Above Normal" (top tercile) = 0 (early spring); "Below Normal" or "Near Normal" = 1 (more winter).
* This mirrors the methodology used by NOAA's own [Grading the Groundhogs analysis](https://www.noaa.gov/heritage/stories/grading-groundhogs).
* Source: NOAA NCEI Climate at a Glance — Pennsylvania Statewide Temperature

#### Temperature Anomaly vs. 30-Year Normal
* The mean temperature from February 2–March 15 was compared to the 1901–2000 climatological mean (~37.5°F for western PA). A positive anomaly (warmer than normal) = 0; zero or negative anomaly = 1.
* This is a continuous rather than tercile-based measure, making it more sensitive to marginal years.
* This is also potentially affected by non-stationarity in the data (e.g. global climate change...)
* Source: NOAA NCEI Climate at a Glance — Pennsylvania Statewide Temperature; [NOAA U.S. Climate Normals 1991–2020](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals)

#### Heating Degree Days (HDD) Accumulation
* Heating Degree Days (HDD) measures how much and for how long the outside air temperature falls below a standard comfort level, typically 65°F. It is used to measure how much heating to necessary to maintain a comfortable temperature inside. 
* Accumulated HDD (base 65°F) for the February 2–March 15 window compared to the climatological long-period mean for Pennsylvania. HDD above the historical mean = 1 (more heating demand = colder = more winter); below = 0. HDD is a standard energy and climate metric that integrates the full temperature distribution rather than relying on the mean alone, and thus captures cold extremes more sensitively than mean anomaly.
* Source: NOAA NCEI Climate at a Glance — Heating Degree Days, Pennsylvania; [NOAA Climate Normals](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals)

#### First Sustained Warm Day (≥50°F for 3+ Consecutive Days)
* This definition defines "early spring" as the arrival of the first sustained warm spell in the post-Groundhog Day window. A run of ≥3 consecutive days with maximum temperature ≥50°F occurring before March 1st = 0 (early spring); first such run occurring on or after March 1, or no qualifying run before April 1, = 1 (more winter). This threshold was chosen to reflect the folk-meteorological intuition underlying the tradition — a brief warm streak that would colloquially signal spring's arrival.
* Of the four definitions, arguably this one most directly aligns with the traditional folklore of Phil's predictions.
* Source: NOAA GHCN-Daily station data — western Pennsylvania composite; primary station USW00094823 (Johnstown/Cambria County area) supplemented with nearby stations for gap-filling.(https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00094823/detail)

#### Composite (Majority Vote)
* A derived variable: coded 1 if ≥3 of the four definitions independently classify the year as "more winter," 0 otherwise. This provides a consensus ground truth less sensitive to any single definition's classification errors.


## Results
#### Confusion Matrix


#### Performance Metrics
| Definition           | N   | Acc   | AUC   | Sens  | Spec  | PPV   | F1    | MCC   | p(binom) |     |
|----------------------|-----|-------|-------|-------|-------|-------|-------|-------|----------|-----|
| CPC Classification   | 124 | 67.7% | 0.623 | 95.8% | 28.8% | 65.1% | 0.775 | 0.346 | 0.00005  | *** |
| Temp Anomaly         | 124 | 69.4% | 0.609 | 93.6% | 28.3% | 68.9% | 0.793 | 0.300 | <0.00001 | *** |
| Heating Degree Days  | 124 | 72.6% | 0.624 | 93.9% | 31.0% | 72.6% | 0.819 | 0.334 | <0.00001 | *** |
| First Warm Day ≥50°F | 124 | 84.7% | 0.710 | 93.9% | 48.0% | 87.7% | 0.907 | 0.478 | <0.00001 | *** |
| Composite            | 124 | 73.4% | 0.628 | 94.0% | 31.7% | 73.6% | 0.825 | 0.343 | <0.00001 | *** |

####  Figures
![](https://github.com/nickmmark/punxsatawney/blob/main/plots/bars.png?raw=true)

![](https://github.com/nickmmark/punxsatawney/blob/main/plots/strip.png?raw=true)

![](https://github.com/nickmmark/punxsatawney/blob/main/plots/rolling.png?raw=true)

## Running the analysis
### Three ways to run the analysis
Use embedded dataset (no files needed)
```
python punxsutawney_phil_analysis.py
```
Load data from the CSV
```
python punxsutawney_phil_analysis.py --csv phil_predictions_1900_2025.csv
```
Custom output dir and rolling window
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

### References
* [Grading the Groundhog](https://www.noaa.gov/heritage/stories/grading-groundhogs), NOAA 2025
