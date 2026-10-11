# Data dictionary: `data/processed/ncr_city_daily.csv`

29,168 rows (one per city per day: 16 cities × 1,823 days, 2021-01-01 to 2025-12-28), 8 columns, no missing values. Every city has its own
numbers; nothing is averaged. Weather values are estimates from the ERA5 weather model (not station readings), in the Asia/Manila time zone.

| Column | Type | Unit | Range (min / median / max) | Definition | Use |
|---|---|---|---|---|---|
| `Date` | date | YYYY-MM-DD | 2021-01-01 to 2025-12-28 | The day the weather was measured ("today"). | Not an input. Puts days in order for the split. |
| `City` | text | name | 16 values | The city or municipality of the National Capital Region: Caloocan, Las Piñas, Makati, Malabon, Mandaluyong, Manila, Marikina, Muntinlupa, Navotas, Parañaque, Pasay, Pasig, Quezon, San Juan, Taguig, Valenzuela. | input (turned into 16 yes/no columns) |
| `HeatIndex_Max_Today` | number | °C | 26.13 / 37.68 / 47.63 | Highest "feels-like" temperature of today in that city. Calculated for every hour from temperature and humidity (US National Weather Service formula); the day's highest hour is kept. | input |
| `Temp_Min` | number | °C | 22.00 / 26.20 / 30.00 | Lowest air temperature of today (2 m above ground), so roughly how warm the night was. | input |
| `Rainfall_Total` | number | mm | 0.0 / 1.3 / 175.1 | Rain that fell during today (sum of the 24 hours). Very skewed: most days are near 0, storm days are huge. | input |
| `Pressure_Mean` | number | hPa | 995.20 / 1010.17 / 1019.31 | Average air pressure at sea level today. Low pressure goes with storms. | input |
| `SolarRadiation_Total` | number | Wh/m² | 478 / 4,830 / 7,551 | Sunshine energy received today (sum of the 24 hours). | input |
| `HeatLevelTomorrow` | text | level | 5 official PAGASA levels (4 occur) | The level of **tomorrow's** highest heat index in that city: `Not Hazardous` (below 27 °C), `Caution` (27 to below 33 °C), `Extreme Caution` (33 to below 42 °C), `Danger` (42 to below 52 °C), `Extreme Danger` (52 °C and above). A value exactly on a boundary goes to the higher level. | **target** (what the model guesses) |

Class counts (all rows): Not Hazardous 40 (0.1 %), Caution 3,954 (13.6 %), Extreme Caution 22,994 (78.8 %), Danger 2,180 (7.5 %), Extreme Danger 0.
Training rows (23,312): 40 / 3,135 / 18,304 / 1,833 / 0. Test rows (5,840): 0 / 806 / 4,687 / 347 / 0.
All 40 Not Hazardous rows are in the training days and Extreme Danger never occurs, so only three levels (Caution, Extreme Caution, Danger) can really be learned and checked.

## The cities really differ

Share of rows whose level tomorrow is `Danger`: Malabon, Navotas, Valenzuela 17.0 %; Marikina 15.1 %; Caloocan 13.7 %; Quezon 10.0 %; Manila, Pasig, Pasay 3.5 %;
Makati, Taguig, Parañaque, Las Piñas, Mandaluyong 3.1 %; Muntinlupa, San Juan 2.0 %. On the same day the hottest and coolest city differ by a median of 1.76 °C in the heat index (at most 5.85 °C).

## How the table was made from the original

`src/make_dataset.py` (the only data step):

1. Take the original file (one row per city per day, 2015 to 2025).
2. Keep the last 5 years: dates from 2021-01-01 on.
3. Keep 8 columns: `Date`, `City`, the 5 weather numbers and `HeatLevelTomorrow`. Nothing is averaged or changed.
4. Checks: the label recomputed from each city's own `HeatIndex_Max_Tomorrow` equals the label in the original (5 official levels; Extreme Danger never occurs: the highest value is 47.63 °C); every day has the same 16 cities; each city's days follow each other with no gap; no missing values.

## What the table does not have (on purpose)

* No `Month` or lag columns ("yesterday's" values): kept out to stay simple; they can be added later if the group wants to test them.
* No years before 2021: the original file has them (2015 to 2020) if the group ever wants more data.

## Limits to remember

* These are **gridded weather-model estimates**, not PAGASA station readings, and the heat index is calculated, so labels are not official PAGASA advisories.
* The 16 cities fall in only **2 weather-model grid cells** (10 cities in one, 6 in the other), so rain, air pressure and sunshine are shared by many cities. The 16 rows of one day are not 16 independent observations. That is why the split and the folds always keep all 16 cities of a day together.
* The 5 inputs describe **today only**. The models predict one day ahead.
* *Danger* is rare (7.5 % of rows) and mostly April to June, so scores for that level vary a lot between folds. *Not Hazardous* appears in only 2 of the 5 check blocks, so `macro_f1` is noisy; use `macro_f1_core3` to compare models.
* Only 5 years: the first fold learns from 246 days (about 8 months), so it covers few hot seasons.
