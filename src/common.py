"""Shared tools for all three models, so they are compared fairly. Everything here is deliberately short.

    from common import *          # in a notebook, first run:  import sys; sys.path.append("../src")

    df = load_data()                                  # one row per day, 7 columns
    train, test = split_train_test(df)                # test = the last 20 % of the days: do not touch until the end
    table = cross_validate(model, train)              # score on 5 time-ordered folds inside the training days
    summarize(table)                                  # mean and spread over the 5 folds
    summarize(persistence_cv(train))                  # the "copy today" rule every model has to beat

A model is any scikit-learn estimator that works on the 5 columns in FEATURES. The target is the text label in TARGET.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

from make_dataset import LEVELS as CLASSES, WEATHER_COLUMNS as FEATURES, to_level

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "ncr_daily.csv"
TARGET = "HeatLevelTomorrow"
DANGER = "Danger"
CORE_LEVELS = ["Caution", "Extreme Caution", "Danger"]   # the 3 levels with hundreds of days; Not Hazardous has only 6, Extreme Danger none
RANDOM_STATE = 42          # use this number for anything random (Random Forest, Gradient Boosting)
TEST_FRACTION = 0.20       # the last 20 % of the days are the test set
GAP_DAYS = 1               # one day is skipped between training and test (and before every fold's check block)
N_FOLDS = 5


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """The daily table, oldest day first."""
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = the first days, test = the last 20 % of the days. We split by date and never shuffle, because the
    models must predict days they have not seen, like a real forecast. One day between the two is skipped: each row's
    label is the NEXT day, so the last training label would otherwise be the first test day's own heat index."""
    n_test = round(len(df) * TEST_FRACTION)
    n_train = len(df) - n_test - GAP_DAYS
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)


def make_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """5 time-ordered folds INSIDE the training days. Fold 1 learns from the earliest days and is checked on the next
    block; every later fold learns from more days and is checked on the block after them. Same folds for every model."""
    return list(TimeSeriesSplit(n_splits=N_FOLDS, gap=GAP_DAYS).split(train))


def score(y_true, y_pred) -> dict:
    """The numbers we report for one block of days.
    macro_f1: the F1 of each class that really occurs in the block, averaged, so the small Danger class counts as much
              as the big Extreme Caution class (accuracy alone would let a model ignore Danger). Careful: Not Hazardous
              has only 6 days in total, so a check block has 0 or 1 of them and this number can jump around.
    macro_f1_core3: the same, but only over Caution, Extreme Caution and Danger, the three levels with hundreds of days.
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
        fit_params=lambda y: {"sample_weight": compute_sample_weight("balanced", y)}
    (inside a Pipeline the name needs the step prefix, for example "model__sample_weight")."""
    X, y = train[FEATURES], train[TARGET]
    rows = []
    for fit_idx, check_idx in make_folds(train):
        m = clone(model)
        extra = fit_params(y.iloc[fit_idx]) if fit_params else {}
        m.fit(X.iloc[fit_idx], y.iloc[fit_idx], **extra)
        rows.append(score(y.iloc[check_idx], m.predict(X.iloc[check_idx])))
    return pd.DataFrame(rows, index=pd.RangeIndex(1, N_FOLDS + 1, name="fold"))


def persistence_predictions(frame: pd.DataFrame) -> pd.Series:
    """The "copy today" rule: predict that tomorrow's level will be the same as today's level."""
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
