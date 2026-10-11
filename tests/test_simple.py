"""Tests for the simple version: the daily file, the split, the folds and the scoring.
Run from the project folder:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common as C  # noqa: E402
import make_dataset as M  # noqa: E402


def tiny_original(days: int = 4, gap_after: int | None = None) -> pd.DataFrame:
    """A made-up original file: 2 cities, known numbers, so the averages can be checked by hand."""
    rows = []
    for d in range(days):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=d + (2 if gap_after is not None and d > gap_after else 0))
        for city, shift in (("A", 0.0), ("B", 2.0)):
            hi_today = 30.0 + 4 * d + shift
            rows.append({"Date": date, "City": city, "HeatIndex_Max_Today": hi_today, "Temp_Min": 24.0 + shift,
                         "Rainfall_Total": 1.0 + d, "Pressure_Mean": 1010.0, "SolarRadiation_Total": 5000.0,
                         "HeatIndex_Max_Tomorrow": hi_today + 4})
    return pd.DataFrame(rows)


class DailyFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = C.load_data()

    def test_seven_columns_one_row_per_day_no_gaps_no_missing(self):
        self.assertEqual(list(self.df.columns), ["Date", *C.FEATURES, "HeatLevelTomorrow"])
        self.assertEqual(self.df.shape, (4015, 7))
        self.assertTrue((self.df["Date"].diff().dropna() == pd.Timedelta(days=1)).all())
        self.assertFalse(self.df.isna().any().any())
        self.assertEqual(set(self.df[C.TARGET]), set(C.CLASSES) - {"Extreme Danger"})     # that level never occurs

    def test_the_file_is_exactly_what_the_script_makes_from_the_original(self):
        rebuilt = M.make_daily(pd.read_csv(M.ORIGINAL))
        pd.testing.assert_frame_equal(self.df, rebuilt, check_dtype=False)

    def test_class_counts(self):
        counts = self.df[C.TARGET].value_counts().reindex(C.CLASSES).fillna(0).astype(int).to_dict()
        self.assertEqual(counts, {"Not Hazardous": 6, "Caution": 540, "Extreme Caution": 3167, "Danger": 302,
                                  "Extreme Danger": 0})

    def test_level_boundaries_go_to_the_higher_level(self):
        s = M.to_level(pd.Series([24.0, 26.99, 27.0, 32.99, 33.0, 41.99, 42.0, 51.99, 52.0, 60.0]))
        self.assertEqual(list(s), ["Not Hazardous", "Not Hazardous", "Caution", "Caution", "Extreme Caution",
                                   "Extreme Caution", "Danger", "Danger", "Extreme Danger", "Extreme Danger"])

    def test_cities_are_averaged_and_the_label_follows_the_average(self):
        d = M.make_daily(tiny_original())
        self.assertEqual(len(d), 4)
        self.assertAlmostEqual(d.loc[0, "HeatIndex_Max_Today"], 31.0)       # mean of 30 and 32
        self.assertAlmostEqual(d.loc[0, "Temp_Min"], 25.0)
        self.assertEqual(d.loc[0, C.TARGET], "Extreme Caution")             # tomorrow's mean is (34 + 36) / 2 = 35

    def test_a_missing_day_is_refused(self):
        with self.assertRaises(AssertionError):
            M.make_daily(tiny_original(days=5, gap_after=1))


class SplitAndFolds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = C.load_data()
        cls.train, cls.test = C.split_train_test(cls.df)

    def test_sizes_and_dates(self):
        self.assertEqual((len(self.train), len(self.test)), (3211, 803))
        self.assertEqual(self.train["Date"].max(), pd.Timestamp("2023-10-16"))
        self.assertEqual(self.test["Date"].min(), pd.Timestamp("2023-10-18"))     # 2023-10-17 is skipped
        self.assertEqual(self.test["Date"].max(), pd.Timestamp("2025-12-28"))
        self.assertEqual(set(self.train["Date"]) & set(self.test["Date"]), set())

    def test_class_counts_in_train_and_test(self):
        # Not Hazardous has 6 days in total, all in the training days, and Extreme Danger never occurs.
        for part, expected in ((self.train, {"Not Hazardous": 6, "Caution": 451, "Extreme Caution": 2523, "Danger": 231,
                                             "Extreme Danger": 0}),
                               (self.test, {"Not Hazardous": 0, "Caution": 89, "Extreme Caution": 643, "Danger": 71,
                                            "Extreme Danger": 0})):
            self.assertEqual(part[C.TARGET].value_counts().reindex(C.CLASSES).fillna(0).astype(int).to_dict(), expected)

    def test_folds_move_forward_in_time_and_stay_inside_training(self):
        folds = C.make_folds(self.train)
        self.assertEqual(len(folds), 5)
        previous_end = -1
        for fit_idx, check_idx in folds:
            fit_dates, check_dates = self.train["Date"].iloc[fit_idx], self.train["Date"].iloc[check_idx]
            self.assertGreater((check_dates.min() - fit_dates.max()).days, C.GAP_DAYS)
            self.assertEqual(set(fit_dates) & set(check_dates), set())
            self.assertGreater(check_idx[0], previous_end)
            previous_end = check_idx[-1]
        self.assertLess(self.train["Date"].iloc[previous_end], self.test["Date"].min())

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

    def test_persistence_copies_todays_level(self):
        df = C.load_data()
        self.assertEqual(list(C.persistence_predictions(df.head(3))), list(M.to_level(df["HeatIndex_Max_Today"].head(3))))


class CrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, _ = C.split_train_test(C.load_data())

    def test_a_placeholder_model_is_scored_on_five_folds(self):
        # DummyClassifier only proves the plumbing works. It is not one of the project's models.
        table = C.cross_validate(DummyClassifier(strategy="most_frequent"), self.train)
        self.assertEqual(table.shape, (5, 5))
        self.assertTrue(((table >= 0) & (table <= 1)).all().all())
        self.assertTrue((table["danger_recall"] == 0).all())

    def test_fit_params_get_the_training_labels_of_each_fold(self):
        seen = []

        def fit_params(y):
            seen.append(len(y))
            return {"sample_weight": np.ones(len(y))}

        C.cross_validate(DecisionTreeClassifier(max_depth=2, random_state=C.RANDOM_STATE), self.train, fit_params)
        self.assertEqual(seen, [535, 1070, 1605, 2140, 2675])

    def test_persistence_summary(self):
        s = C.summarize(C.persistence_cv(self.train))
        self.assertAlmostEqual(s.loc["macro_f1", "mean"], 0.541, places=3)
        self.assertAlmostEqual(s.loc["macro_f1_core3", "mean"], 0.678, places=3)
        self.assertAlmostEqual(s.loc["accuracy", "mean"], 0.837, places=3)

    def test_final_test_is_the_only_place_that_reads_the_test_days(self):
        train, test = C.split_train_test(C.load_data())
        scores = C.final_test(DummyClassifier(strategy="most_frequent"), train, test)
        self.assertAlmostEqual(scores["accuracy"], 643 / 803, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
