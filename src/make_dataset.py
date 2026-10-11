"""Step 1, the only data step: turn the original 16-city file into one simple row per day for Metro Manila.

    original:  data/original/ph_heat_index_next_day_16cities.csv   (one row per city per day, 64,240 rows)
    result:    data/processed/ncr_daily.csv                        (one row per day, 4,015 rows)

What happens, in plain words:
  1. For every date, average the 16 cities. They share almost the same weather, so one row per day loses very little.
  2. Tomorrow's heat index is the average of the cities' "HeatIndex_Max_Tomorrow" (the next day's highest heat index).
  3. The heat index is turned into the 5 official PAGASA heat levels (the labels the models must predict).
  4. Only 5 inputs are kept, so the finished table has 7 columns: Date, the 5 inputs and the label.

Run it with:  python src/make_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data" / "original" / "ph_heat_index_next_day_16cities.csv"
PROCESSED = ROOT / "data" / "processed" / "ncr_daily.csv"

# The 5 official PAGASA heat levels. A value exactly on a boundary goes to the higher level (33.00 is "Extreme Caution").
#   below 27         Not Hazardous     (very rare here: 6 days)
#   27 to below 33   Caution
#   33 to below 42   Extreme Caution
#   42 to below 52   Danger
#   52 and above     Extreme Danger    (never happens in this data: the highest daily value is 45.52)
BINS = [-np.inf, 27.0, 33.0, 42.0, 52.0, np.inf]
LEVELS = ["Not Hazardous", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]

# The 5 inputs. They were picked on the training days only: today's heat index has by far the strongest link to
# tomorrow's (Spearman 0.84), and the other four each add something beyond it and tell a different story:
#   Temp_Min             how warm last night was (heat that carries over)
#   Rainfall_Total       a rainy or stormy day cools the next one
#   Pressure_Mean        low pressure means storms
#   SolarRadiation_Total how sunny today was
WEATHER_COLUMNS = ["HeatIndex_Max_Today", "Temp_Min", "Rainfall_Total", "Pressure_Mean", "SolarRadiation_Total"]
OUTPUT_COLUMNS = ["Date"] + WEATHER_COLUMNS + ["HeatLevelTomorrow"]


def to_level(heat_index: pd.Series) -> pd.Series:
    """Heat index (degrees C, 2 decimals) -> one of the 5 level names."""
    return pd.cut(heat_index, bins=BINS, labels=LEVELS, right=False).astype(str)


def make_daily(original: pd.DataFrame) -> pd.DataFrame:
    """One row per day: the mean over the cities of every column we keep."""
    df = original.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cities_per_day = df.groupby("Date")["City"].nunique()
    assert (cities_per_day == cities_per_day.iloc[0]).all(), "every day must have the same cities"

    daily = df.groupby("Date")[WEATHER_COLUMNS + ["HeatIndex_Max_Tomorrow"]].mean().round(2).reset_index()
    daily["HeatLevelTomorrow"] = to_level(daily["HeatIndex_Max_Tomorrow"])

    # Check: tomorrow's heat index must be the next day's "today" heat index (rounding can differ by 0.01).
    steps = daily["Date"].diff().dropna()
    assert (steps == pd.Timedelta(days=1)).all(), "the days must follow each other without gaps"
    next_day = daily["HeatIndex_Max_Today"].shift(-1)
    assert ((next_day - daily["HeatIndex_Max_Tomorrow"]).abs().dropna() <= 0.011).all()
    assert daily[OUTPUT_COLUMNS].notna().all().all(), "no missing values expected"
    return daily[OUTPUT_COLUMNS]       # "HeatIndex_Max_Tomorrow" was only needed to make the label


def main() -> int:
    original = pd.read_csv(ORIGINAL)
    daily = make_daily(original)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    daily.assign(Date=daily["Date"].dt.strftime("%Y-%m-%d")).to_csv(PROCESSED, index=False, lineterminator="\n")
    print(f"wrote {PROCESSED.relative_to(ROOT)}: {len(daily):,} rows x {daily.shape[1]} columns, "
          f"{daily['Date'].min().date()} to {daily['Date'].max().date()}")
    print(daily["HeatLevelTomorrow"].value_counts().reindex(LEVELS).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
