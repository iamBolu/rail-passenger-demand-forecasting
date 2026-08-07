# Rail Passenger Demand Forecasting

A machine learning project that forecasts daily rail passenger demand using historical ridership data.

## Project Overview

This project analyzes daily rail ridership data and builds forecasting models to predict future rail passenger demand.

The project compares a traditional time-series model, ARIMA, with a machine learning model, Gradient Boosting, to determine which approach performs better.

## Dataset

The dataset contains daily public transportation ridership information, including:

- Service date
- Day type
- Bus boardings
- Rail boardings
- Total rides

The target variable is `rail_boardings`.

## Project Workflow

1. Data cleaning
2. Exploratory data analysis
3. Trend analysis
4. Seasonal decomposition
5. Feature engineering
6. Train/test split
7. ARIMA modeling
8. Gradient Boosting modeling
9. Model evaluation
10. Feature importance analysis

## Feature Engineering

The following features were created to help the machine learning model identify demand patterns:

- Year
- Month
- Day of week
- Day of year
- Previous day's demand
- Demand 7 days ago
- Demand 30 days ago
- 7-day rolling average
- 30-day rolling average

## Models

### ARIMA

ARIMA was used as the traditional time-series benchmark. It learns patterns directly from historical rail demand.

### Gradient Boosting

Gradient Boosting was trained using calendar features, lagged demand, and rolling averages to learn relationships between these variables and passenger demand.

## Results

| Model             |     MAE |    RMSE |
| ----------------- | ------: | ------: |
| ARIMA             | 294,818 | 370,057 |
| Gradient Boosting | 103,704 | 156,742 |

Gradient Boosting performed substantially better than ARIMA.

Compared with ARIMA, Gradient Boosting achieved:

- 64.82% lower MAE
- 57.64% lower RMSE

## Feature Importance

The most important feature for the Gradient Boosting model was `day_of_week`, accounting for approximately 70% of the model's feature importance.

Other important features included:

- `lag_1`
- `rolling_7`
- `lag_7`
- `day_of_year`

This indicates that weekly patterns and recent passenger demand were particularly useful for predicting rail ridership.

## Example Prediction

For one test observation:

- Predicted rail boardings: 378,277
- Actual rail boardings: 361,156
- Difference: 17,121

## Project Structure

```text
rail-passenger-demand-forecasting/
│
├── data/
│   ├── Ridership.csv
│   └── model_ready.csv
│
├── models/
│   ├── arima_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
│
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```
