"""
Forecast daily rail passenger demand from the trained Gradient Boosting model.

The model needs nine features for any day it predicts. Four come from the
calendar and are always available. The other five - three lags and two rolling
averages - are built from actual boardings on preceding days, so a forecast for
day D requires real history up to D-1.

Usage
-----
    python src/forecast.py                        # next day after the history ends
    python src/forecast.py --date 2017-10-01      # a specific date
    python src/forecast.py --date 2021-11-01 --days 7
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "ridership_cleaned.csv"
MODEL_PATH = ROOT / "models" / "gradient_boosting_model.pkl"

# Order matters - the model was trained on these columns in this order.
FEATURES = [
    "year", "month", "day_of_week", "day_of_year",
    "lag_1", "lag_7", "lag_30", "rolling_7", "rolling_30",
]

MIN_HISTORY_DAYS = 30  # rolling_30 needs the 30 days before the target date


def load_history(path=HISTORY_PATH):
    """Load actual daily boardings, indexed by date."""
    df = pd.read_csv(path, parse_dates=["service_date"])
    df = df.sort_values("service_date").set_index("service_date")
    return df["rail_boardings"].astype(float)


def load_model(path=MODEL_PATH):
    """Load the trained Gradient Boosting model."""
    return joblib.load(path)


def build_features(history, target_date):
    """
    Build the nine model features for a single date.

    `history` is a Series of actual boardings indexed by date. Every feature is
    derived only from days strictly before `target_date`, which is the same
    constraint applied during training - the target never appears in its own
    features.
    """
    target_date = pd.Timestamp(target_date)
    past = history[history.index < target_date]  # Only days before the target

    if len(past) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_DAYS} days of history before "
            f"{target_date.date()}, found {len(past)}."
        )

    def lag(days):
        """Actual boardings exactly `days` before the target date."""
        wanted = target_date - pd.Timedelta(days=days)
        if wanted not in past.index:
            raise ValueError(f"No history for {wanted.date()}, needed for lag_{days}.")
        return past.loc[wanted]

    return pd.DataFrame(
        [{
            "year": target_date.year,
            "month": target_date.month,
            "day_of_week": target_date.dayofweek,  # Monday=0, Sunday=6
            "day_of_year": target_date.dayofyear,
            "lag_1": lag(1),
            "lag_7": lag(7),
            "lag_30": lag(30),
            "rolling_7": past.iloc[-7:].mean(),    # Mean of the 7 days before target
            "rolling_30": past.iloc[-30:].mean(),  # Mean of the 30 days before target
        }],
        index=[target_date],
    )[FEATURES]


def forecast(history, model, start_date=None, days=1):
    """
    Forecast `days` consecutive days starting at `start_date`.

    Beyond the first day there is no actual value to build the next day's lags
    from, so each prediction is appended to the history and used as though it
    were real. Errors therefore compound as the horizon grows - the same
    limitation that causes the ARIMA baseline to decay to a flat line. Treat
    anything past a few days as indicative only.
    """
    if start_date is None:
        start_date = history.index.max() + pd.Timedelta(days=1)
    start_date = pd.Timestamp(start_date)

    working = history.copy()  # Grows as predictions are appended
    rows = []

    for step in range(days):
        date = start_date + pd.Timedelta(days=step)
        prediction = float(model.predict(build_features(working, date))[0])

        actual = float(history.loc[date]) if date in history.index else None
        rows.append({
            "date": date.date(),
            "day": date.strftime("%A"),
            "predicted": prediction,
            "actual": actual,
            "error": abs(prediction - actual) if actual is not None else None,
            "basis": "actual history" if step == 0 else f"{step} predicted day(s)",
        })

        working.loc[date] = prediction  # Feed the prediction forward
        working = working.sort_index()

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Forecast daily rail passenger demand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", help="First date to forecast (YYYY-MM-DD). "
                                       "Defaults to the day after the history ends.")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of consecutive days to forecast (default: 1).")
    args = parser.parse_args()

    history = load_history()
    model = load_model()

    print(f"History: {history.index.min().date()} to {history.index.max().date()} "
          f"({len(history):,} days)\n")

    try:
        results = forecast(history, model, args.date, args.days)
    except ValueError as err:
        raise SystemExit(f"Cannot forecast: {err}")

    for row in results.itertuples():
        line = f"{row.date}  {row.day:<9}  predicted {row.predicted:>10,.0f}"
        if row.actual is not None:
            pct = row.error / row.actual * 100
            line += f"   actual {row.actual:>10,.0f}   off by {row.error:>8,.0f} ({pct:.1f}%)"
        print(line)

    if args.days > 1:
        print("\nNote: only the first day is based purely on actual history. Later "
              "days build on earlier predictions, so error compounds.")

    known = results[results["actual"].notna()]
    if not known.empty:
        print(f"\nMean absolute error over {len(known)} day(s) with known actuals: "
              f"{known['error'].mean():,.0f}")


if __name__ == "__main__":
    main()
