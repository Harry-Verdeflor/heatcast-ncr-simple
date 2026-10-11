# Data dictionary: `data/processed/ncr_daily.csv`

4,015 rows (one per day, 2015-01-01 to 2025-12-28), 7 columns, no missing values. Every number is the **average of the 16
cities of Metro Manila** for that day. Weather values are estimates from the ERA5 weather model (not station readings), in the Asia/Manila time zone.

| Column | Type | Unit | Range (min / median / max) | Definition | Use |
|---|---|---|---|---|---|
| `Date` | date | YYYY-MM-DD | 2015-01-01 to 2025-12-28 | The day the weather was measured ("today"). | Not an input. Puts days in order for the split. |
| `HeatIndex_Max_Today` | number | °C | 24.18 / 37.61 / 45.52 | Highest "feels-like" temperature of today. Calculated for every hour from temperature and humidity (US National Weather Service formula); the day's highest hour is kept; then averaged over the cities. | input |
| `Temp_Min` | number | °C | 21.16 / 26.16 / 29.60 | Lowest air temperature of today (2 m above ground), so roughly how warm the night was. | input |
| `Rainfall_Total` | number | mm | 0.00 / 1.10 / 162.70 | Rain that fell during today (sum of the 24 hours). Very skewed: most days are near 0, storm days are huge. | input |
| `Pressure_Mean` | number | hPa | 995.24 / 1010.29 / 1020.26 | Average air pressure at sea level today. Low pressure goes with storms. | input |
| `SolarRadiation_Total` | number | Wh/m² | 333.88 / 4942.38 / 7763.75 | Sunshine energy received today (sum of the 24 hours). | input |
| `HeatLevelTomorrow` | text | level | 3 values | The level of **tomorrow's** highest heat index: `Caution or below` (below 33 °C), `Extreme Caution` (33 to below 42 °C), `Danger` (42 °C and above). A value exactly on a boundary goes to the higher level. | **target** (what the model guesses) |

Class counts: Caution or below 546 (13.6 %), Extreme Caution 3,167 (78.9 %), Danger 302 (7.5 %).

## How the table was made from the original

`src/make_dataset.py` (the only data step):

1. Take the original file (one row per city per day).
2. For each date, average the 16 cities (all 16 are present on every date; checked).
3. Tomorrow's heat index = the average of the cities' `HeatIndex_Max_Tomorrow` (the next day's highest heat index; checked to equal the next day's `HeatIndex_Max_Today`).
4. Turn tomorrow's heat index into the 3 levels. The two lowest official levels (Not Hazardous, Caution) are joined, and Danger and Extreme Danger are joined (Extreme Danger never occurs: the highest value in the original is 48.16 °C).
5. Keep `Date`, the 5 inputs and the label.

## What the table does not have (on purpose)

* No city column: the cities are averaged. On the same day the hottest and coolest city differ by a median of 1.75 °C in the heat index (95 % of days under 3.4 °C, at most 5.9 °C), mostly because the 16 cities fall in only 2 weather-model grid cells. That is not independent information, but the average does hide it.
* No `Month` or lag columns ("yesterday's" values): kept out to stay simple; they can be added later if the group wants to test them.
* Averaging cities lowers the extremes a little: the highest daily value is 45.52 °C here against 48.16 °C in a single city.

## Limits to remember

* These are **gridded weather-model estimates**, not PAGASA station readings, and the heat index is calculated, so labels are not official PAGASA advisories.
* The 5 inputs describe **today only**. The models predict one day ahead.
* *Danger* is rare (7.5 % of days) and mostly April to June, so scores for that level vary a lot between folds.
