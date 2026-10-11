"""Tests for the simple version: the data file, the split, the folds and the scoring.
Run from the project folder:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common as C  # noqa: E402
import make_dataset as M  # noqa: E402


def dense(matrix) -> np.ndarray:
    """The preprocessing may return a sparse matrix (the city columns); turn it into a plain array for checking."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def tiny_original(first="2020-12-30", days=5, wrong_label=False, skip_day=None) -> pd.DataFrame:
    """A made-up original file: 2 cities, known numbers, so the steps can be checked by hand."""
    rows = []
    for d in range(days):
        if skip_day is not None and d == skip_day:
            continue
        date = pd.Timestamp(first) + pd.Timedelta(days=d)
        for city, shift in (("A", 0.0), ("B", 6.0)):
            hi_tomorrow = 30.0 + 2 * d + shift
            rows.append({"Date": date, "City": city, "HeatIndex_Max_Today": hi_tomorrow - 2, "Temp_Min": 24.0 + shift / 6,
                         "Rainfall_Total": 1.0 + d, "Pressure_Mean": 1010.0, "SolarRadiation_Total": 5000.0,
                         "HeatIndex_Max_Tomorrow": hi_tomorrow,
                         "HeatLevelTomorrow": "Danger" if wrong_label else M.to_level(pd.Series([hi_tomorrow])).iloc[0]})
    return pd.DataFrame(rows)


class DataFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = C.load_data()

    def test_eight_columns_16_cities_every_day_no_gaps_no_missing(self):
        self.assertEqual(list(self.df.columns), ["Date", "City", *C.NUMERIC, "HeatLevelTomorrow"])
        self.assertEqual(self.df.shape, (29168, 8))
        self.assertEqual(self.df["Date"].nunique(), 1823)
        self.assertEqual((self.df["Date"].min(), self.df["Date"].max()), (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-28")))
        self.assertTrue((self.df.groupby("Date")["City"].nunique() == 16).all())
        for _, one_city in self.df.groupby("City"):
            self.assertTrue((one_city["Date"].diff().dropna() == pd.Timedelta(days=1)).all())
        self.assertFalse(self.df.isna().any().any())

    def test_the_file_is_exactly_what_the_script_makes_from_the_original(self):
        rebuilt = M.make_city_daily(pd.read_csv(M.ORIGINAL))
        pd.testing.assert_frame_equal(self.df, rebuilt, check_dtype=False)

    def test_class_counts(self):
        counts = self.df[C.TARGET].value_counts().reindex(C.CLASSES).fillna(0).astype(int).to_dict()
        self.assertEqual(counts, {"Not Hazardous": 40, "Caution": 3954, "Extreme Caution": 22994, "Danger": 2180,
                                  "Extreme Danger": 0})

    def test_cities_keep_their_own_numbers_and_levels(self):
        # nothing is averaged: on most days the 16 cities do not all have the same heat index
        different = (self.df.groupby("Date")["HeatIndex_Max_Today"].nunique() > 1).mean()
        self.assertGreater(different, 0.99)
        danger_share = self.df.groupby("City")[C.TARGET].apply(lambda s: (s == "Danger").mean())
        self.assertGreater(danger_share["Malabon"], 0.15)        # one of the hottest cities ...
        self.assertLess(danger_share["San Juan"], 0.03)          # ... against one of the coolest

    def test_level_boundaries_go_to_the_higher_level(self):
        s = M.to_level(pd.Series([24.0, 26.99, 27.0, 32.99, 33.0, 41.99, 42.0, 51.99, 52.0, 60.0]))
        self.assertEqual(list(s), ["Not Hazardous", "Not Hazardous", "Caution", "Caution", "Extreme Caution",
                                   "Extreme Caution", "Danger", "Danger", "Extreme Danger", "Extreme Danger"])

    def test_only_the_last_five_years_are_kept_and_numbers_are_untouched(self):
        out = M.make_city_daily(tiny_original())
        self.assertEqual(out["Date"].min(), pd.Timestamp("2021-01-01"))            # 2020-12-30 and 2020-12-31 are dropped
        self.assertEqual(len(out), 6)                                                # 3 days x 2 cities
        row = out[(out["Date"] == "2021-01-01") & (out["City"] == "B")].iloc[0]
        self.assertAlmostEqual(row["HeatIndex_Max_Today"], 38.0)                     # B on 2021-01-01: tomorrow 40, today 38
        self.assertEqual(row["HeatLevelTomorrow"], "Extreme Caution")                # 40 is in 33 to below 42

    def test_a_wrong_label_or_a_missing_day_is_refused(self):
        with self.assertRaises(AssertionError):
            M.make_city_daily(tiny_original(wrong_label=True))
        with self.assertRaises(AssertionError):
            M.make_city_daily(tiny_original(first="2021-01-01", days=5, skip_day=2))


class SplitAndFolds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = C.load_data()
        cls.train, cls.test = C.split_train_test(cls.df)

    def test_sizes_and_dates(self):
        self.assertEqual((len(self.train), len(self.test)), (23312, 5840))
        self.assertEqual((self.train["Date"].nunique(), self.test["Date"].nunique()), (1457, 365))
        self.assertEqual(self.train["Date"].max(), pd.Timestamp("2024-12-27"))
        self.assertEqual(self.test["Date"].min(), pd.Timestamp("2024-12-29"))        # 2024-12-28 is skipped
        self.assertEqual(self.test["Date"].max(), pd.Timestamp("2025-12-28"))
        self.assertEqual(set(self.train["Date"]) & set(self.test["Date"]), set())

    def test_all_16_cities_of_a_day_are_on_the_same_side(self):
        for part in (self.train, self.test):
            self.assertTrue((part.groupby("Date")["City"].nunique() == 16).all())

    def test_class_counts_in_train_and_test(self):
        # Not Hazardous has 40 rows in total, all in the training days; Extreme Danger never occurs.
        for part, expected in ((self.train, {"Not Hazardous": 40, "Caution": 3135, "Extreme Caution": 18304, "Danger": 1833,
                                             "Extreme Danger": 0}),
                               (self.test, {"Not Hazardous": 0, "Caution": 806, "Extreme Caution": 4687, "Danger": 347,
                                            "Extreme Danger": 0})):
            self.assertEqual(part[C.TARGET].value_counts().reindex(C.CLASSES).fillna(0).astype(int).to_dict(), expected)

    def test_folds_move_forward_in_time_by_whole_days_and_stay_inside_training(self):
        folds = C.make_folds(self.train)
        self.assertEqual(len(folds), 5)
        previous_end = pd.Timestamp("2000-01-01")
        for fit_idx, check_idx in folds:
            fit, check = self.train.iloc[fit_idx], self.train.iloc[check_idx]
            self.assertGreater((check["Date"].min() - fit["Date"].max()).days, C.GAP_DAYS)
            self.assertEqual(set(fit["Date"]) & set(check["Date"]), set())
            self.assertTrue((check.groupby("Date")["City"].nunique() == 16).all())      # a day is never cut in half
            self.assertGreater(check["Date"].min(), previous_end)
            previous_end = check["Date"].max()
        self.assertLess(previous_end, self.test["Date"].min())

    def test_every_check_block_has_the_three_core_levels(self):
        for _, check_idx in C.make_folds(self.train):
            self.assertTrue(set(C.CORE_LEVELS) <= set(self.train[C.TARGET].iloc[check_idx]))


class Scoring(unittest.TestCase):
    def test_macro_f1_ignores_a_class_that_is_not_in_the_block(self):
        y = ["Extreme Caution", "Danger", "Extreme Caution"]
        self.assertAlmostEqual(C.score(y, y)["macro_f1"], 1.0)

    def test_core3_f1_only_looks_at_caution_extreme_caution_and_danger(self):
        y = ["Not Hazardous", "Caution", "Extreme Caution", "Danger"]
        s = C.score(y, y)
        self.assertAlmostEqual(s["macro_f1"], 1.0)
        self.assertAlmostEqual(s["macro_f1_core3"], 1.0)
        wrong = ["Caution", "Caution", "Extreme Caution", "Danger"]          # the Not Hazardous day is called Caution
        self.assertAlmostEqual(C.score(y, wrong)["macro_f1"], (0 + 2 / 3 + 1 + 1) / 4)
        self.assertAlmostEqual(C.score(y, wrong)["macro_f1_core3"], (2 / 3 + 1 + 1) / 3)

    def test_danger_recall_and_precision(self):
        y_true = ["Danger"] * 4 + ["Extreme Caution"] * 4
        y_pred = ["Danger"] * 3 + ["Extreme Caution"] + ["Danger"] + ["Extreme Caution"] * 3
        s = C.score(y_true, y_pred)
        self.assertAlmostEqual(s["danger_recall"], 0.75)
        self.assertAlmostEqual(s["danger_precision"], 0.75)

    def test_persistence_copies_todays_level_of_each_city(self):
        df = C.load_data()
        self.assertEqual(list(C.persistence_predictions(df.head(3))), list(M.to_level(df["HeatIndex_Max_Today"].head(3))))


class Preprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = C.split_train_test(C.load_data())

    def test_scaling_and_city_columns(self):
        pre = C.make_preprocess(scale=True).fit(self.train[C.FEATURES])
        out = dense(pre.transform(self.train[C.FEATURES]))
        self.assertEqual(out.shape, (len(self.train), 5 + 16))                       # 5 numbers + 16 yes/no city columns
        self.assertTrue(np.allclose(out[:, :5].mean(axis=0), 0, atol=1e-6))
        self.assertTrue(np.allclose(out[:, :5].std(axis=0), 1, atol=1e-6))

    def test_without_scaling_the_numbers_are_unchanged(self):
        out = dense(C.make_preprocess(scale=False).fit_transform(self.train[C.FEATURES]))
        np.testing.assert_allclose(out[:, :5], self.train[C.NUMERIC].to_numpy())

    def test_the_scaler_learns_from_training_days_only(self):
        pre = C.make_preprocess(scale=True).fit(self.train[C.FEATURES])
        scaler = pre.named_transformers_["numbers"]
        self.assertAlmostEqual(scaler.mean_[0], self.train["HeatIndex_Max_Today"].mean(), places=6)
        self.assertNotAlmostEqual(scaler.mean_[0], pd.concat([self.train, self.test])["HeatIndex_Max_Today"].mean(), places=3)

    def test_an_unknown_city_is_ignored_not_an_error(self):
        pre = C.make_preprocess().fit(self.train[C.FEATURES])
        row = self.test[C.FEATURES].head(1).copy()
        row["City"] = "Atlantis"
        out = dense(pre.transform(row))
        self.assertEqual(float(out[0, 5:].sum()), 0.0)


class CrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = C.split_train_test(C.load_data())

    def test_a_placeholder_model_is_scored_on_five_folds(self):
        # DummyClassifier only proves the plumbing works. It is not one of the project's models.
        model = make_pipeline(C.make_preprocess(), DummyClassifier(strategy="most_frequent"))
        table = C.cross_validate(model, self.train)
        self.assertEqual(table.shape, (5, 5))
        self.assertTrue(((table >= 0) & (table <= 1)).all().all())
        self.assertTrue((table["danger_recall"] == 0).all())

    def test_fit_params_get_the_training_labels_of_each_fold(self):
        seen = []

        def fit_params(y):
            seen.append(len(y))
            return {"decisiontreeclassifier__sample_weight": np.ones(len(y))}

        model = make_pipeline(C.make_preprocess(scale=False), DecisionTreeClassifier(max_depth=2, random_state=C.RANDOM_STATE))
        C.cross_validate(model, self.train, fit_params)
        self.assertEqual(seen, [3936, 7808, 11680, 15552, 19424])

    def test_persistence_summary(self):
        s = C.summarize(C.persistence_cv(self.train))
        self.assertAlmostEqual(s.loc["macro_f1", "mean"], 0.620, places=3)
        self.assertAlmostEqual(s.loc["macro_f1_core3", "mean"], 0.679, places=3)
        self.assertAlmostEqual(s.loc["accuracy", "mean"], 0.846, places=3)

    def test_final_test_is_the_only_place_that_reads_the_test_days(self):
        model = make_pipeline(C.make_preprocess(), DummyClassifier(strategy="most_frequent"))
        scores = C.final_test(model, self.train, self.test)
        self.assertAlmostEqual(scores["accuracy"], 4687 / 5840, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
