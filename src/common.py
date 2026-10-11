"""Shared tools for all three models, so they are compared fairly. Everything here is deliberately short.

    from common import *          # in a notebook, first run:  import sys; sys.path.append("../src")

    df = load_data()                                              # one row per city per day, 8 columns
    train, test = split_train_test(df)                            # test = the last 20 % of the days: do not touch until the end
    model = make_pipeline(make_preprocess(scale=True), LogisticRegression(max_iter=1000))
    table = cross_validate(model, train)                          # score on 5 time-ordered folds inside the training days
    summarize(table)                                              # mean and spread over the 5 folds
    summarize(persistence_cv(train))                              # the "copy today" rule every model has to beat

A model is any scikit-learn estimator that works on the columns in FEATURES (the 5 weather numbers and the city name).
Put make_preprocess() first in the pipeline: it scales the numbers (only if scale=True) and turns the city into 0/1 columns.
The target is the text label in TARGET.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from make_dataset import LEVELS as CLASSES, WEATHER_COLUMNS as NUMERIC, to_level

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "ncr_city_daily.csv"
FEATURES = NUMERIC + ["City"]
TARGET = "HeatLevelTomorrow"
DANGER = "Danger"
CORE_LEVELS = ["Caution", "Extreme Caution", "Danger"]   # the 3 levels with many days; Not Hazardous is rare, Extreme Danger never happens
RANDOM_STATE = 42          # use this number for anything random (Random Forest, Gradient Boosting)
TEST_FRACTION = 0.20       # the last 20 % of the days are the test set
GAP_DAYS = 1               # one day is skipped between training and test (and before every fold's check block)
N_FOLDS = 5


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """One row per city per day, oldest day first (the 16 cities of a day sit together)."""
    return pd.read_csv(path, parse_dates=["Date"]).sort_values(["Date", "City"]).reset_index(drop=True)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = the first days, test = the last 20 % of the days. We split by DATE and never shuffle, so the models must
    predict days they have not seen, like a real forecast, and all 16 cities of a day stay on the same side (they have
    nearly the same weather, so a shuffle would let the model peek). One day between the two is skipped: each row's label
    is the NEXT day, so the last training label would otherwise be the first test day's own heat index."""
    days = np.sort(df["Date"].unique())
    n_test = round(len(days) * TEST_FRACTION)
    n_train = len(days) - n_test - GAP_DAYS
    train = df[df["Date"].isin(days[:n_train])].reset_index(drop=True)
    test = df[df["Date"].isin(days[-n_test:])].reset_index(drop=True)
    return train, test


def make_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """5 time-ordered folds INSIDE the training days. Fold 1 learns from the earliest days and is checked on the next
    block of days; every later fold learns from more days and is checked on the block after them. The folds are cut by
    day, so the 16 cities of a day are never split. Same folds for every model. Returns row positions of `train`."""
    days = np.sort(train["Date"].unique())
    day_of_row = np.searchsorted(days, train["Date"].to_numpy())
    folds = []
    for fit_days, check_days in TimeSeriesSplit(n_splits=N_FOLDS, gap=GAP_DAYS).split(days):
        folds.append((np.flatnonzero(np.isin(day_of_row, fit_days)), np.flatnonzero(np.isin(day_of_row, check_days))))
    return folds


def make_preprocess(scale: bool = True) -> ColumnTransformer:
    """Get the table ready for a model: the 5 weather numbers are scaled (scale=True, needed by Logistic Regression;
    trees do not need it) and the city becomes 16 yes/no columns. Put it first in a pipeline, so it learns its numbers
    from the training days of each fold only."""
    return ColumnTransformer([("numbers", StandardScaler() if scale else "passthrough", NUMERIC),
                              ("city", OneHotEncoder(handle_unknown="ignore"), ["City"])])


def score(y_true, y_pred) -> dict:
    """The numbers we report for one block of days.
    macro_f1: the F1 of each class that really occurs in the block, averaged, so the small Danger class counts as much
              as the big Extreme Caution class (accuracy alone would let a model ignore Danger). Careful: Not Hazardous
              is rare, so a check block has few of them and this number can jump around.
    macro_f1_core3: the same, but only over Caution, Extreme Caution and Danger, the three levels with many days.
              Steadier, so it is the better number for comparing models.
    danger_recall: of the real Danger days, the share the model caught (the costly mistake is a missed Danger day).
    danger_precision: of the days the model called Danger, the share that really were."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = np.unique(y_true)
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[DANGER], zero_division=0)
    return {"macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
            "macro_f1_core3": f1_score(y_true, y_pred, labels=CORE_LEVELS, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "danger_recall": float(r[0]), "danger_precision": float(p[0])}


def cross_validate(model, train: pd.DataFrame, fit_params=None) -> pd.DataFrame:
    """Fit a fresh copy of `model` on each fold's training days and score it on that fold's check block.
    Returns one row of scores per fold. `fit_params` is optional: a function that takes the training labels and
    returns extra arguments for fit, for example the sample weights of Gradient Boosting:
        fit_params=lambda y: {"gradientboostingclassifier__sample_weight": compute_sample_weight("balanced", y)}
    (inside a pipeline the name needs the step name in front, here the one make_pipeline gives)."""
    X, y = train[FEATURES], train[TARGET]
    rows = []
    for fit_idx, check_idx in make_folds(train):
        m = clone(model)
        extra = fit_params(y.iloc[fit_idx]) if fit_params else {}
        m.fit(X.iloc[fit_idx], y.iloc[fit_idx], **extra)
        rows.append(score(y.iloc[check_idx], m.predict(X.iloc[check_idx])))
    return pd.DataFrame(rows, index=pd.RangeIndex(1, N_FOLDS + 1, name="fold"))


def persistence_predictions(frame: pd.DataFrame) -> pd.Series:
    """The "copy today" rule: predict that tomorrow's level will be the same as today's level (in the same city)."""
    return to_level(frame["HeatIndex_Max_Today"])


def persistence_cv(train: pd.DataFrame) -> pd.DataFrame:
    """The "copy today" rule scored on the same folds as the models. A model is only useful if it beats this."""
    y, guess = train[TARGET], persistence_predictions(train)
    rows = [score(y.iloc[check_idx], guess.iloc[check_idx]) for _, check_idx in make_folds(train)]
    return pd.DataFrame(rows, index=pd.RangeIndex(1, N_FOLDS + 1, name="fold"))


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread (standard deviation) of every score over the 5 folds."""
    return pd.DataFrame({"mean": table.mean(), "sd": table.std(ddof=1)}).round(3)


def final_test(model, train: pd.DataFrame, test: pd.DataFrame, fit_params=None) -> dict:
    """THE FINAL CHECK. Fit on all training days and score on the test days. Do this once, for the model the group
    picked, and only after all the choosing is finished."""
    m = clone(model)
    extra = fit_params(train[TARGET]) if fit_params else {}
    m.fit(train[FEATURES], train[TARGET], **extra)
    return score(test[TARGET], m.predict(test[FEATURES]))
