# Rail Passenger Demand Forecasting

Forecasting daily rail passenger demand on the Chicago Transit Authority network
using twenty years of historical ridership data.

## Motivation

Transit agencies decide how much service to run — how frequently trains arrive,
how many crew to roster, when a station closes — based on how many passengers
they expect. Getting that wrong is costly in both directions: under-predict and
you get overcrowding, over-predict and you run empty carriages and pay for crew
you did not need.

This project builds and evaluates a model for the forecast that sits underneath
those decisions.

## Dataset

Chicago Transit Authority daily ridership, January 2001 to November 2021.

| Column | Description |
|---|---|
| `service_date` | Date of service |
| `day_type` | W = weekday, A = Saturday, U = Sunday/holiday |
| `bus` | Daily bus boardings |
| `rail_boardings` | **Daily rail boardings — the target variable** |
| `total_rides` | Bus + rail |

The target is a single system-wide daily total. The dataset contains no route
or time-of-day breakdown.

**Row counts:** 7,701 raw → 7,639 after removing 62 duplicate rows → 7,609 after
dropping the first 30 rows, which have insufficient history for the 30-day lag.

## How to run

```bash
git clone https://github.com/iamBolu/rail-passenger-demand-forecasting.git
cd rail-passenger-demand-forecasting
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order — each writes the file the next one reads:

1. `01_data_exploration.ipynb` — cleaning → `data/ridership_cleaned.csv`
2. `02_exploratory_data_analysis.ipynb` — EDA, decomposition, feature engineering → `data/model_ready.csv`
3. `03_modeling.ipynb` — training, evaluation, cross-validation → `models/`

The ARIMA fit in notebook 03 takes a few minutes on 6,087 observations.

To inspect the saved results without re-running anything:

```bash
python -c "import pandas as pd; print(pd.read_csv('models/model_comparison.csv'))"
```

## Making a forecast

`src/forecast.py` loads the trained model and produces a prediction for a given
date, rebuilding the nine features from historical boardings.

```bash
python src/forecast.py                          # next day after the history ends
python src/forecast.py --date 2017-10-01        # a specific date
python src/forecast.py --date 2018-03-05 --days 7
```

```text
$ python src/forecast.py --date 2018-03-05 --days 7
History: 2001-01-01 to 2021-11-30 (7,639 days)

2018-03-05  Monday     predicted    685,278   actual    686,228   off by      950 (0.1%)
2018-03-06  Tuesday    predicted    709,448   actual    723,274   off by   13,826 (1.9%)
2018-03-07  Wednesday  predicted    717,910   actual    731,992   off by   14,082 (1.9%)
2018-03-08  Thursday   predicted    726,540   actual    734,447   off by    7,907 (1.1%)
2018-03-09  Friday     predicted    734,099   actual    712,092   off by   22,007 (3.1%)
2018-03-10  Saturday   predicted    437,868   actual    394,014   off by   43,854 (11.1%)
2018-03-11  Sunday     predicted    319,041   actual    286,948   off by   32,093 (11.2%)
```

Only the first day is built purely from actual history. Because a forecast for
day D needs actual boardings up to D-1, each subsequent day is built on earlier
predictions and error compounds — 0.1% on day one, roughly 11% by day seven.
This is the same mechanism that causes the ARIMA baseline to decay to a flat
line over a long horizon, and it is why the model is only trustworthy one day
ahead unless fresh actuals are supplied.

## Feature engineering

A date is a unique value, so there is no repeating pattern in it to learn from.
Nine features were extracted, covering the cycles that actually drive travel
behaviour:

**Calendar — what kind of day is this?**
`year`, `month`, `day_of_week`, `day_of_year`

**Lags — past values**
`lag_1` (previous day), `lag_7` (same weekday last week), `lag_30` (30 days back)

**Rolling means — current demand level**
`rolling_7`, `rolling_30`

### Avoiding target leakage

Rolling features are computed as `.shift(1).rolling(n).mean()` — shifted *before*
the window is applied, so the current day is never inside its own average.

Without the shift, the target leaks into its own feature. The model would score
well in testing and fail in deployment, because the current day's boardings are
not available at the moment the prediction is made. Every feature satisfies the
constraint that its value would genuinely be known at prediction time.

## Validation

**Chronological 80/20 split**, not random:

| | Range | Days |
|---|---|---:|
| Train | 2001-01-31 → 2017-09-30 | 6,087 |
| Test | 2017-10-01 → 2021-11-30 | 1,522 |

Random splitting is invalid here for two reasons: it allows training on data that
postdates the test set, and with lag features adjacent rows leak answers — a
held-out day's value appears directly in the following day's `lag_1`.

**Cross-validation** uses `TimeSeriesSplit` with five expanding windows, so each
fold trains only on data preceding its test period:

| Fold | Test period | MAE | RMSE |
|---:|---|---:|---:|
| 1 | 2004-07 → 2008-01 | 31,175 | 53,337 |
| 2 | 2008-01 → 2011-07 | 40,327 | 60,631 |
| 3 | 2011-07 → 2014-12 | 65,630 | 91,773 |
| 4 | 2014-12 → 2018-06 | 35,383 | 56,402 |
| 5 | 2018-06 → 2021-11 | 123,332 | 179,471 |

Mean MAE across all folds: **59,169**. Across folds 1–4: **43,129**.

Folds 1–4 are stable, which indicates the result does not depend on one
convenient split. Fold 5 is the only failure, and it is the only fold whose test
window contains the 2020 structural break.

## Models

**ARIMA(7, 1, 1)** — statsmodels. Univariate baseline; sees only the ridership
series, with no access to the engineered features.

**GradientBoostingRegressor** — scikit-learn.
`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `random_state=42`.

## Results

| Model | MAE | RMSE |
|---|---:|---:|
| ARIMA | 294,818 | 370,057 |
| Gradient Boosting | **103,704** | **156,742** |

Gradient Boosting reduced MAE by 64.8% and RMSE by 57.6%.

### Accuracy by period

The headline figure blends a working model with a structural break, so error is
reported by regime:

| Period | Days | Mean demand | GB MAE | GB error % | ARIMA MAE |
|---|---:|---:|---:|---:|---:|
| Normal (Oct 2017 – Feb 2020) | 882 | 605,426 | **35,046** | **5.8%** | 137,136 |
| COVID crash (Mar – Dec 2020) | 306 | 139,118 | 208,502 | 149.9% | 551,466 |
| Recovery (2021) | 334 | 214,320 | 189,000 | 88.2% | 476,077 |
| Whole test period | 1,522 | 425,846 | 103,704 | 24.4% | 294,818 |

**Under normal conditions the model forecasts daily demand to within
approximately 5.8%.**

### Accuracy by forecast horizon

The headline comparison is not like-for-like: ARIMA forecasts all 1,522 days in a
single call with no new observations, while Gradient Boosting receives `lag_1` on
every test row and is therefore effectively predicting one day ahead throughout.

| Horizon | ARIMA MAE | GB MAE |
|---|---:|---:|
| 7 days | 18,521 | 17,538 |
| 14 days | 35,982 | 36,441 |
| 30 days | 51,422 | 30,833 |
| 90 days | 101,262 | 34,610 |
| 365 days | 121,121 | 32,598 |
| 1,522 days | 294,818 | 103,704 |

Over a one-week horizon the models are level, and at two weeks ARIMA is
marginally ahead. The 65% gap is largely an artefact of forecast length rather
than of model quality — ARIMA feeds on its own predictions and decays to a flat
line near 690,000 by roughly day 100, while Gradient Boosting's error is
approximately constant across horizons. A like-for-like comparison would refit
ARIMA on a rolling one-day-ahead basis.

## Feature importance

| Feature | Importance |
|---|---:|
| `day_of_week` | **70.2%** |
| `lag_1` | 10.4% |
| `rolling_7` | 5.9% |
| `lag_7` | 5.1% |
| `day_of_year` | 4.4% |
| `rolling_30` | 2.8% |
| `year` | 0.9% |
| `lag_30` | 0.27% |
| `month` | 0.04% |

Weekly seasonality dominates: a typical Thursday is around 734,000 boardings
against roughly 287,000 on a Sunday. Commuting is a weekly habit, and that cycle
outweighs every other signal.

The two weakest features were both attempting to capture a monthly cycle, which
is not a real behavioural rhythm. `lag_30` is also misaligned by construction —
30 days is not a multiple of 7, so it lands two weekdays out: every Monday's
`lag_30` falls on a Saturday. `lag_28` would preserve weekday alignment.

### Example prediction

Sunday 1 October 2017 — predicted **378,277**, actual **361,156**, difference
**17,121** (4.7%).

## Limitations

1. **The 2020 structural break sits inside the test window.** COVID was kept in
   deliberately: a test set should simulate reality, and cutting the data at 2019
   would have improved the headline number by removing the only genuinely hard
   part of the evaluation. Accuracy is therefore reported by regime.

2. **Error remains elevated through 2021** even as ridership recovers, because
   the lag features carry pre-pandemic demand forward. In deployment this model
   would require drift monitoring and a retraining trigger — it does not signal
   that it has gone stale, it continues predicting the world that used to exist.

3. **The ARIMA comparison is not like-for-like**, as quantified above.

4. **The data is daily, not hourly.** The model answers "how busy will next
   Thursday be", not "how busy will 8am be".

5. **Not deployed.** `src/forecast.py` provides a command-line interface to the
   trained model, but there is no serving layer, live data feed, or scheduled
   retraining. This is an analysis and modelling project, not a production
   system.

6. **Forecasts are only reliable one day ahead.** The lag features require
   actual boardings up to the previous day. Beyond that the script feeds its own
   predictions forward and error compounds, as shown above.

## Project structure

```text
rail-passenger-demand-forecasting/
├── data/
│   ├── Ridership.csv                        raw CTA data
│   ├── ridership_cleaned.csv                after cleaning
│   └── model_ready.csv                      with engineered features
├── models/
│   ├── arima_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── cv_results.csv
│   ├── error_by_period.csv
│   └── error_by_horizon.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_modeling.ipynb
├── src/
│   └── forecast.py                          CLI: date → prediction
├── README.md
└── requirements.txt
```

## Stack

Python 3.13 · pandas · NumPy · scikit-learn · statsmodels · matplotlib · seaborn
· joblib · Jupyter

## Next steps

- Add public holiday and weather features
- Refit ARIMA on a rolling basis for a like-for-like comparison
- Replace `lag_30` with `lag_28` to preserve weekday alignment
- Extend to a 7-day-ahead horizon, since crew rostering happens weeks in advance
