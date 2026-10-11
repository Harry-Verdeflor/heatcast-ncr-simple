"""Step 1, the only data step: from the original file to the simple table the models use.

    original:  data/original/ph_heat_index_next_day_16cities.csv   (16 cities, 2015 to 2025, 26 columns, 64,240 rows)
    result:    data/processed/ncr_city_daily.csv                   (16 cities, 2021 to 2025, 8 columns, 29,168 rows)

What happens, in plain words:
  1. Keep the last 5 years only: 2021-01-01 to 2025-12-28.
  2. Keep 8 columns: the date, the city, 5 weather numbers about today, and tomorrow's PAGASA heat level.
  3. Every city keeps its own numbers. Nothing is averaged, because the heat index really differs from city to city.

Run it with:  python src/make_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data" / "original" / "ph_heat_index_next_day_16cities.csv"
PROCESSED = ROOT / "data" / "processed" / "ncr_city_daily.csv"
START_DATE = "2021-01-01"      # the last 5 years: 2021, 2022, 2023, 2024 and 2025 (the data ends on 2025-12-28)

# The 5 official PAGASA heat levels. A value exactly on a boundary goes to the higher level (33.00 is "Extreme Caution").
#   below 27         Not Hazardous     (rare)
#   27 to below 33   Caution
#   33 to below 42   Extreme Caution
#   42 to below 52   Danger
#   52 and above     Extreme Danger    (never happens in this data: the highest heat index is 47.63)
BINS = [-np.inf, 27.0, 33.0, 42.0, 52.0, np.inf]
LEVELS = ["Not Hazardous", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]

# The 5 weather inputs. They were picked on the training days only: today's heat index has by far the strongest link
# to tomorrow's, and the other four each add something beyond it and tell a different story:
#   Temp_Min             how warm last night was (heat that carries over)
#   Rainfall_Total       a rainy or stormy day cools the next one
#   Pressure_Mean        low pressure means storms
#   SolarRadiation_Total how sunny today was
WEATHER_COLUMNS = ["HeatIndex_Max_Today", "Temp_Min", "Rainfall_Total", "Pressure_Mean", "SolarRadiation_Total"]
OUTPUT_COLUMNS = ["Date", "City"] + WEATHER_COLUMNS + ["HeatLevelTomorrow"]


def to_level(heat_index: pd.Series) -> pd.Series:
    """Heat index (degrees C, 2 decimals) -> one of the 5 level names."""
    return pd.cut(heat_index, bins=BINS, labels=LEVELS, right=False).astype(str)


def make_city_daily(original: pd.DataFrame) -> pd.DataFrame:
    """One row per city per day for the last 5 years, with the 8 columns."""
    df = original.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= START_DATE].sort_values(["Date", "City"]).reset_index(drop=True)

    # Checks. The label is recomputed from each city's own heat index for tomorrow and must match the original label.
    assert (to_level(df["HeatIndex_Max_Tomorrow"]) == df["HeatLevelTomorrow"].astype(str)).all(), "label mismatch"
    cities_per_day = df.groupby("Date")["City"].nunique()
    assert (cities_per_day == cities_per_day.iloc[0]).all(), "every day must have the same cities"
    for _, one_city in df.groupby("City"):
        assert (one_city["Date"].diff().dropna() == pd.Timedelta(days=1)).all(), "the days must follow each other"
    assert df[OUTPUT_COLUMNS].notna().all().all(), "no missing values expected"
    return df[OUTPUT_COLUMNS]


def main() -> int:
    daily = make_city_daily(pd.read_csv(ORIGINAL))
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    daily.assign(Date=daily["Date"].dt.strftime("%Y-%m-%d")).to_csv(PROCESSED, index=False, lineterminator="\n")
    print(f"wrote {PROCESSED.relative_to(ROOT)}: {len(daily):,} rows x {daily.shape[1]} columns, "
          f"{daily['City'].nunique()} cities, {daily['Date'].nunique():,} days "
          f"({daily['Date'].min().date()} to {daily['Date'].max().date()})")
    print(daily["HeatLevelTomorrow"].value_counts().reindex(LEVELS).fillna(0).astype(int).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
