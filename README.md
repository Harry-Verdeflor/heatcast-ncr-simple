# HeatCast NCR (simple version)

**Question:** can we guess how hot tomorrow will *feel* in Metro Manila, using only today's weather?

The answer is one of the five official PAGASA **heat levels**: *Not Hazardous*, *Caution*, *Extreme Caution*, *Danger* or *Extreme Danger*. We compare three
traditional machine learning models, **Logistic Regression**, **Random Forest** and **Gradient Boosting**, and put the best one in a small web app.

This is the simplified restart of the project: **one table, 8 columns, the last 5 years, every city on its own row.** If a word is new to you, see
[`documentation/plain_language_guide.md`](documentation/plain_language_guide.md).

## The data in one picture

`data/processed/ncr_city_daily.csv` has **29,168 rows** (16 cities × 1,823 days, 2021-01-01 to 2025-12-28) and **8 columns**:

| Column | What it is | Role |
|---|---|---|
| `Date` | the day | only used to put days in order |
| `City` | one of the 16 cities of Metro Manila | input (becomes 16 yes/no columns) |
| `HeatIndex_Max_Today` | today's highest "feels-like" temperature in that city (°C) | input |
| `Temp_Min` | today's lowest temperature, i.e. last night (°C) | input |
| `Rainfall_Total` | rain today (mm) | input |
| `Pressure_Mean` | average air pressure today (hPa) | input |
| `SolarRadiation_Total` | how much sun today (Wh/m²) | input |
| `HeatLevelTomorrow` | tomorrow's PAGASA level in that city: Not Hazardous / Caution / Extreme Caution / Danger / Extreme Danger | **the answer the model must guess** |

**Every city keeps its own numbers.** Nothing is averaged, because the heat index really differs from city to city: the
hottest cities (Malabon, Navotas, Valenzuela) have *Danger* tomorrow on 17 % of days, the coolest (Muntinlupa, San Juan) on only 2 %.
Rows: 40 *Not Hazardous*, 3,954 *Caution*, 22,994 *Extreme Caution*, 2,180 *Danger*, 0 *Extreme Danger*.
**Not Hazardous (40 rows) and Extreme Danger (never) are too rare for a model to learn.** The levels that count are Caution, Extreme Caution and Danger.

## Which folder is which

| Folder / file | What is in it |
|---|---|
| `data/original/` | **the original data** (one row per city per day, 26 columns, 64,240 rows, 2015 to 2025) and `SOURCE.md` saying where it came from |
| `data/processed/ncr_city_daily.csv` | **the simple dataset** made from the original by `src/make_dataset.py` |
| `src/make_dataset.py` | the one data step: original → simple dataset |
| `src/common.py` | shared tools: load the data, split train/test, 5 folds, preprocessing, scoring. **All three models use it.** |
| `tests/` | automatic checks (`python -m unittest discover -s tests -v`) |
| `notebooks/` | one notebook per person (see below) |
| `models/` | saved models (later) |
| `app/` | the Streamlit app (later) |
| `documentation/` | plain-language guide and the data dictionary |
| `paper/` | the final paper (later) |

## How the work goes (7 steps) and who does what

1. **Define the problem**: done (this README).
2. **Describe the data**: done (`documentation/data_dictionary.md`, `data/original/SOURCE.md`).
3. **Explore the data (EDA)**: `notebooks/01_eda.ipynb`, Marc Lawrence David. At least 5 useful charts, each with a written finding (one of them: the Danger share per city).
4. **Prepare the data**: done and kept tiny: the split, the 5 folds, the scaling and the city columns, and the scoring are in `src/common.py`.
5. **Train the three models**, each on the same split and folds:
   `notebooks/02_logistic_regression.ipynb` (Harry Verdeflor), `notebooks/03_random_forest.ipynb` (Edrian Tala),
   `notebooks/04_gradient_boosting.ipynb` (Marc Lawrence David).
6. **Compare and pick the best model**, then check it **once** on the test days.
7. **Deploy** it in a small Streamlit app (`app/`).

## Quick start

```bash
pip install -r requirements.txt
python src/make_dataset.py                 # rebuilds data/processed/ncr_city_daily.csv (optional, it is already there)
python -m unittest discover -s tests -v    # 24 checks, a few seconds
```

At the top of a notebook (the notebooks live in `notebooks/`):

```python
import sys; sys.path.append("../src")
from common import *
from sklearn.linear_model import LogisticRegression        # or RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

df = load_data()
train, test = split_train_test(df)          # test = the last 20 % of the days. Do not look at it until the very end.
model = make_pipeline(make_preprocess(scale=True), LogisticRegression(max_iter=1000))   # scale=False for the two tree models
scores = cross_validate(model, train)       # 5 time-ordered folds inside the training days
summarize(scores)                           # mean and spread
summarize(persistence_cv(train))            # the "copy today" rule. Your model has to beat it.
```

`make_preprocess()` scales the 5 weather numbers (only Logistic Regression needs that) and turns `City` into 16 yes/no columns.
It goes first in the pipeline, so it learns its numbers from the training days of each fold only.

## The rules that keep the comparison fair

* **Split by date, never at random.** The last 20 % of the days (2024-12-29 to 2025-12-28, 365 days) are the test days. The first 1,457 days (2021-01-01 to 2024-12-27) are for training. One day between them is skipped. All 16 cities of a day are on the same side, because they share much of the weather and a random split would let the model peek.
* **Choose everything with the 5 folds inside the training days.** The folds are cut by whole days, forward in time. Never use the test days to choose anything.
* **Use `src/common.py` for every score**, and `RANDOM_STATE = 42` for anything random.
* **Check the test days once**, for the one model the group picks: `final_test(model, train, test)`.
* **Main score: macro F1** (it counts the small *Danger* class as much as the big one). Because *Not Hazardous* is rare (it appears in only 2 of the 5 check blocks), `macro_f1` can jump around; `macro_f1_core3` (only Caution, Extreme Caution, Danger) is steadier. Also report accuracy, Danger recall and precision, and a confusion matrix.

The "copy today" rule (predict that tomorrow has the same level as today, in the same city) gets macro F1 **0.620 ± 0.102** (**0.679 ± 0.051** over the three core levels), accuracy 0.846 and Danger recall 0.518 on the 5 folds (0.675, 0.824 and 0.597 on the test days). It is hard to beat: today's heat index already says a lot about tomorrow. A model is only useful if it does better.

Good to know: the first fold learns from only 246 days (about 8 months), so it is the weakest practice round.

## Where the data comes from

Hourly weather from the [Open-Meteo](https://open-meteo.com/) Historical Weather API (ERA5 reanalysis, CC BY 4.0, non-commercial use),
for the 16 cities of the National Capital Region. The heat index is calculated from hourly temperature and humidity with the
US National Weather Service formula. Details in [`data/original/SOURCE.md`](data/original/SOURCE.md). The code that downloaded it,
with the raw responses, is in the earlier repository [`keyyum/heatcast-ph`](https://github.com/keyyum/heatcast-ph).

*Weather data by Open-Meteo.com, containing modified Copernicus Climate Change Service (ERA5) information. City list: Philippine Statistics Authority PSGC via the `psgc` package; boundaries OCHA/HDX (CC BY-IGO).*
