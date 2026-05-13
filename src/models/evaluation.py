"""Evaluation utilities for chronological macroeconomic forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplitWindow:
    """Chronological train/test split boundaries."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def verify_chronological_order(frame: pd.DataFrame, year_column: str = "year") -> None:
    """Raise when a frame is not sorted chronologically."""

    years = frame[year_column].to_numpy()
    if len(years) and not np.array_equal(years, np.sort(years)):
        raise ValueError(f"Rows are not sorted by {year_column}.")


def expanding_window_splits(
    n_observations: int,
    *,
    n_splits: int = 3,
    test_size: int = 3,
    min_train_size: int = 8,
) -> list[TimeSeriesSplitWindow]:
    """Create expanding-window splits without shuffling or future leakage."""

    if n_observations < min_train_size + test_size:
        return []

    max_splits = (n_observations - min_train_size) // test_size
    split_count = min(n_splits, max_splits)
    initial_train_size = n_observations - split_count * test_size

    splits: list[TimeSeriesSplitWindow] = []
    for fold in range(split_count):
        train_start = 0
        train_end = initial_train_size + fold * test_size
        test_start = train_end
        test_end = test_start + test_size
        splits.append(
            TimeSeriesSplitWindow(
                fold=fold + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return splits


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Root mean squared error."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Mean absolute error."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def mape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Mean absolute percentage error, excluding zero actual values."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    nonzero = actual_array != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual_array[nonzero] - predicted_array[nonzero]) / actual_array[nonzero])) * 100)


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Compute the required forecast evaluation metrics."""

    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
    }


def summarize_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate forecast metrics by country and model."""

    if predictions.empty:
        return pd.DataFrame(
            columns=["country", "model", "n_predictions", "rmse", "mae", "mape"]
        )

    rows: list[dict[str, float | int | str]] = []
    for (country, model), group in predictions.groupby(["country", "model"]):
        metrics = evaluate_predictions(group["actual"], group["predicted"])
        rows.append(
            {
                "country": country,
                "model": model,
                "n_predictions": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["country", "rmse", "model"]).reset_index(drop=True)
