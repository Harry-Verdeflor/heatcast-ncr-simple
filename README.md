# HeatCast NCR (simple version)

**Question:** can we guess how hot tomorrow will *feel* in Metro Manila, using only today's weather?

The answer is one of three **heat levels**: *Caution or below*, *Extreme Caution* or *Danger*. We compare three
traditional machine learning models, **Logistic Regression**, **Random Forest** and **Gradient Boosting**, and put the best one in a small web app.

This is the simplified restart of the project: **one table, 7 columns, one row per day.** If a word is new to you, see
[`documentation/plain_language_guide.md`](documentation/plain_language_guide.md).

## The data in one picture

`data/processed/ncr_daily.csv` has **4,015 rows (days, 2015-01-01 to 2025-12-28)** and **7 columns**:

| Column | What it is | Role |
|---|---|---|
| `Date` | the day | only used to put days in order |
| `HeatIndex_Max_Today` | today's highest "feels-like" temperature (°C) | input |
| `Temp_Min` | today's lowest temperature, i.e. last night (°C) | input |
| `Rainfall_Total` | rain today (mm) | input |
| `Pressure_Mean` | average air pressure today (hPa) | input |
| `SolarRadiation_Total` | how much sun today (Wh/m²) | input |
| `HeatLevelTomorrow` | tomorrow's level: Caution or below / Extreme Caution / Danger | **the answer the model must guess** |

Each row is Metro Manila's weather for one day (the average of its 16 cities, which have very similar weather) and
the heat level of the **next** day. Days: 546 *Caution or below*, 3,167 *Extreme Caution*, 302 *Danger*.

## Which folder is which

| Folder / file | What is in it |
|---|---|
| `data/original/` | **the original data** (one row per city per day, 26 columns, 64,240 rows) and `SOURCE.md` saying where it came from |
| `data/processed/ncr_daily.csv` | **the simple dataset** made from the original by `src/make_dataset.py` |
| `src/make_dataset.py` | the one data step: original → simple dataset |
| `src/common.py` | shared tools: load the data, split train/test, 5 folds, scoring. **All three models use it.** |
| `tests/` | automatic checks (`python -m unittest discover -s tests -v`) |
| `notebooks/` | one notebook per person (see below) |
| `models/` | saved models (later) |
| `app/` | the Streamlit app (later) |
| `documentation/` | plain-language guide and the data dictionary |
| `paper/` | the final paper (later) |

## How the work goes (7 steps) and who does what

1. **Define the problem**: done (this README).
2. **Describe the data**: done (`documentation/data_dictionary.md`, `data/original/SOURCE.md`).
3. **Explore the data (EDA)**: `notebooks/01_eda.ipynb`, Marc Lawrence David. At least 5 useful charts, each with a written finding.
4. **Prepare the data**: done and kept tiny: the split, the 5 folds and the scoring are in `src/common.py`.
5. **Train the three models**, each on the same split and folds:
   `notebooks/02_logistic_regression.ipynb` (Harry Verdeflor), `notebooks/03_random_forest.ipynb` (Edrian Tala),
   `notebooks/04_gradient_boosting.ipynb` (Marc Lawrence David).
6. **Compare and pick the best model**, then check it **once** on the test days.
7. **Deploy** it in a small Streamlit app (`app/`).

## Quick start

```bash
pip install -r requirements.txt
python src/make_dataset.py                 # rebuilds data/processed/ncr_daily.csv (optional, it is already there)
python -m unittest discover -s tests -v    # 17 checks, about a second
```

At the top of a notebook (the notebooks live in `notebooks/`):

```python
import sys; sys.path.append("../src")
from common import *
from sklearn.linear_model import LogisticRegression        # or RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = load_data()
train, test = split_train_test(df)          # test = the last 20 % of the days. Do not look at it until the very end.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_validate(model, train)       # 5 time-ordered folds inside the training days
summarize(scores)                           # mean and spread
summarize(persistence_cv(train))            # the "copy today" rule. Your model has to beat it.
```

## The rules that keep the comparison fair

* **Split by date, never at random.** The last 20 % of the days (2023-10-18 to 2025-12-28, 803 days) are the test days. The first 3,211 days (to 2023-10-16) are for training. One day between them is skipped.
* **Choose everything with the 5 folds inside the training days.** Never use the test days to choose anything.
* **Use `src/common.py` for every score**, and `RANDOM_STATE = 42` for anything random.
* **Check the test days once**, for the one model the group picks: `final_test(model, train, test)`.
* **Main score: macro F1** (it counts the small *Danger* class as much as the big one). Also report accuracy, Danger recall and precision, and a confusion matrix.

The "copy today" rule (predict that tomorrow has the same level as today) gets macro F1 **0.681 ± 0.031** on the 5 folds. A model is only useful if it does better.

## Where the data comes from

Hourly weather from the [Open-Meteo](https://open-meteo.com/) Historical Weather API (ERA5 reanalysis, CC BY 4.0, non-commercial use),
for the 16 cities of the National Capital Region. The heat index is calculated from hourly temperature and humidity with the
US National Weather Service formula. Details in [`data/original/SOURCE.md`](data/original/SOURCE.md). The code that downloaded it,
with the raw responses, is in the earlier repository [`keyyum/heatcast-ph`](https://github.com/keyyum/heatcast-ph).

*Weather data by Open-Meteo.com, containing modified Copernicus Climate Change Service (ERA5) information. City list: Philippine Statistics Authority PSGC via the `psgc` package; boundaries OCHA/HDX (CC BY-IGO).*
