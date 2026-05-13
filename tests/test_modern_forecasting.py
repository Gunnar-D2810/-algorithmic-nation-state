"""Tests for modern forecasting utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.evaluation import expanding_window_splits, mae, mape, rmse
from src.models.model_registry import get_model_registry
from src.models.modern_forecasting import add_lagged_features, write_forecast_outputs


class ModernForecastingTests(unittest.TestCase):
    """Basic contract tests for the modern forecasting layer."""

    def test_model_registry_contains_expected_names(self) -> None:
        registry = get_model_registry()
        for name in [
            "arima",
            "sarimax",
            "random_forest",
            "prophet",
            "lightgbm",
            "lstm",
            "optional_nbeats",
            "optional_nhits",
            "optional_tft_or_patchtst",
        ]:
            self.assertIn(name, registry)

    def test_lag_feature_generation_uses_previous_year(self) -> None:
        frame = pd.DataFrame(
            {
                "country": ["USA", "USA", "USA"],
                "year": [2000, 2001, 2002],
                "GDP_GROWTH": [1.0, 2.0, 3.0],
                "INFLATION": [4.0, 5.0, 6.0],
            }
        )
        lagged = add_lagged_features(
            frame,
            features=["INFLATION"],
        )
        self.assertTrue(pd.isna(lagged.loc[0, "GDP_GROWTH_L1"]))
        self.assertEqual(lagged.loc[1, "GDP_GROWTH_L1"], 1.0)
        self.assertEqual(lagged.loc[2, "INFLATION_L1"], 5.0)

    def test_chronological_split_boundaries(self) -> None:
        splits = expanding_window_splits(
            17,
            n_splits=3,
            test_size=3,
            min_train_size=8,
        )
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assertLessEqual(split.train_end, split.test_start)
            self.assertLess(split.test_start, split.test_end)

    def test_metrics(self) -> None:
        actual = pd.Series([1.0, 2.0, 4.0])
        predicted = pd.Series([1.0, 3.0, 2.0])
        self.assertAlmostEqual(rmse(actual, predicted), (5 / 3) ** 0.5)
        self.assertAlmostEqual(mae(actual, predicted), 1.0)
        self.assertAlmostEqual(mape(actual, predicted), ((0.0 + 0.5 + 0.5) / 3) * 100)

    def test_output_file_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_dir = Path(tmpdir)
            predictions = pd.DataFrame(
                [{"country": "USA", "model": "naive", "year": 2020, "actual": 1.0, "predicted": 1.1}]
            )
            empty = pd.DataFrame()
            failures = pd.DataFrame(columns=["country", "model", "reason"])
            warnings = pd.DataFrame(columns=["country", "model", "rmse", "naive_rmse", "warning"])
            write_forecast_outputs(
                tables_dir=tables_dir,
                predictions=predictions,
                fold_metrics=empty,
                modern_comparison=empty,
                all_model_comparison=empty,
                failures=failures,
                naive_warnings=warnings,
            )
            self.assertTrue((tables_dir / "modern_forecast_predictions.csv").exists())
            self.assertTrue((tables_dir / "modern_forecast_comparison.csv").exists())
            self.assertTrue((tables_dir / "all_model_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
