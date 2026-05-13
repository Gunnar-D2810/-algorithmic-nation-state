"""Shared utilities for exploratory modern forecasting benchmarks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.models.evaluation import (
    evaluate_predictions,
    expanding_window_splits,
    summarize_model_metrics,
    verify_chronological_order,
)
from src.models.lightgbm_model import forecast_lightgbm
from src.models.lstm_model import forecast_lstm
from src.models.model_registry import (
    MODEL_REGISTRY,
    ForecastModelSkipped,
    dependency_status,
)
from src.models.prophet_model import forecast_prophet

LOGGER = logging.getLogger(__name__)

TARGET_INDICATOR = "GDP_GROWTH"
FEATURE_INDICATORS = [
    "INFLATION",
    "UNEMPLOYMENT",
    "EXPORTS_PERCENT_GDP",
    "IMPORTS_PERCENT_GDP",
    "FDI_NET_INFLOWS",
    "HIGH_TECH_EXPORTS",
    "MILITARY_EXPENDITURE",
]


def load_macro_panel(panel_path: Path) -> pd.DataFrame:
    """Load and validate the cleaned long-form macro panel."""

    panel = pd.read_csv(panel_path)
    required_columns = {"country", "year", "indicator", "value"}
    missing_columns = required_columns.difference(panel.columns)
    if missing_columns:
        raise ValueError(f"Macro panel is missing required columns: {missing_columns}")

    duplicates = panel.duplicated(subset=["country", "year", "indicator"], keep=False)
    if duplicates.any():
        duplicate_rows = panel.loc[duplicates, ["country", "year", "indicator"]]
        raise ValueError(
            "Macro panel contains duplicate rows: "
            f"{duplicate_rows.head(20).to_dict(orient='records')}"
        )

    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    return panel.dropna(subset=["year"]).copy()


def macro_panel_to_wide(panel: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Pivot a long indicator panel to country-year rows."""

    wide = (
        panel.loc[panel["indicator"].isin(indicators)]
        .pivot_table(
            index=["country", "year"],
            columns="indicator",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide.sort_values(["country", "year"]).reset_index(drop=True)


def add_lagged_features(
    country_frame: pd.DataFrame,
    *,
    target: str = TARGET_INDICATOR,
    features: list[str] | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Create lagged predictors using only previous-year values."""

    features = FEATURE_INDICATORS if features is None else features
    lagged = country_frame.sort_values("year").copy()
    for column in [target, *features]:
        if column in lagged.columns:
            # Assumption: annual GDP growth at year t is predicted using values
            # observed no later than year t-1. Missing source values stay missing.
            lagged[f"{column}_L{lag}"] = lagged[column].shift(lag)
    return lagged


def lag_feature_columns(
    *,
    target: str = TARGET_INDICATOR,
    features: list[str] | None = None,
    lag: int = 1,
) -> list[str]:
    """Return lag feature names for machine-learning style models."""

    features = FEATURE_INDICATORS if features is None else features
    return [f"{target}_L{lag}", *[f"{feature}_L{lag}" for feature in features]]


def run_naive_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    train_start_year: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run last-observed-value validation for one country."""

    return _run_target_only_country(
        country,
        country_frame,
        model_name="naive",
        train_start_year=train_start_year,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        forecast_fn=lambda train, test_years: pd.Series(
            [float(train[TARGET_INDICATOR].iloc[-1])] * len(test_years),
            dtype=float,
        ),
    )


def run_prophet_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    train_start_year: int,
    n_splits: int,
    test_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Prophet validation for one country."""

    spec = MODEL_REGISTRY["prophet"]
    _ensure_dependencies("prophet")
    return _run_target_only_country(
        country,
        country_frame,
        model_name="prophet",
        train_start_year=train_start_year,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=spec.min_train_size,
        forecast_fn=lambda train, test_years: forecast_prophet(
            train,
            test_years,
            min_train_size=spec.min_train_size,
        ),
    )


def run_lightgbm_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    train_start_year: int,
    n_splits: int,
    test_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run LightGBM validation for one country."""

    spec = MODEL_REGISTRY["lightgbm"]
    _ensure_dependencies("lightgbm")
    return _run_lagged_feature_country(
        country,
        country_frame,
        model_name="lightgbm",
        train_start_year=train_start_year,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=spec.min_train_size,
        forecast_fn=lambda train_x, train_y, test_x: forecast_lightgbm(
            train_x,
            train_y,
            test_x,
            min_train_size=spec.min_train_size,
        ),
    )


def run_lstm_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    train_start_year: int,
    n_splits: int,
    test_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run LSTM validation for one country when TensorFlow is available."""

    spec = MODEL_REGISTRY["lstm"]
    _ensure_dependencies("lstm")
    return _run_lagged_feature_country(
        country,
        country_frame,
        model_name="lstm",
        train_start_year=train_start_year,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=spec.min_train_size,
        forecast_fn=lambda train_x, train_y, test_x: forecast_lstm(
            train_x,
            train_y,
            test_x,
            min_train_size=spec.min_train_size,
        ),
    )


def run_neuralforecast_placeholder(
    country: str,
    model_name: str,
    country_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Record NeuralForecast models as skipped unless conditions are adequate."""

    spec = MODEL_REGISTRY[model_name]
    available, dependency_reason = dependency_status(model_name)
    usable_target_rows = int(country_frame[TARGET_INDICATOR].notna().sum())
    reasons: list[str] = []
    if not available:
        reasons.append(dependency_reason)
    if usable_target_rows < spec.min_train_size:
        reasons.append(
            f"Only {usable_target_rows} non-missing target rows; {spec.min_train_size} required."
        )
    reasons.append("NeuralForecast benchmarks are not enabled in this first modern layer.")
    raise ForecastModelSkipped(" ".join(reasons))


def run_model_for_country(
    model_name: str,
    country: str,
    country_frame: pd.DataFrame,
    *,
    train_start_year: int,
    n_splits: int,
    test_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dispatch one model for one country."""

    runners = {
        "naive": lambda: run_naive_country(
            country,
            country_frame,
            train_start_year=train_start_year,
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=MODEL_REGISTRY["naive"].min_train_size,
        ),
        "prophet": lambda: run_prophet_country(
            country,
            country_frame,
            train_start_year=train_start_year,
            n_splits=n_splits,
            test_size=test_size,
        ),
        "lightgbm": lambda: run_lightgbm_country(
            country,
            country_frame,
            train_start_year=train_start_year,
            n_splits=n_splits,
            test_size=test_size,
        ),
        "lstm": lambda: run_lstm_country(
            country,
            country_frame,
            train_start_year=train_start_year,
            n_splits=n_splits,
            test_size=test_size,
        ),
        "optional_nbeats": lambda: run_neuralforecast_placeholder(
            country, "optional_nbeats", country_frame
        ),
        "optional_nhits": lambda: run_neuralforecast_placeholder(
            country, "optional_nhits", country_frame
        ),
        "optional_tft_or_patchtst": lambda: run_neuralforecast_placeholder(
            country, "optional_tft_or_patchtst", country_frame
        ),
    }
    if model_name not in runners:
        raise ValueError(f"Modern forecasting runner does not handle model: {model_name}")
    return runners[model_name]()


def add_naive_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add naive RMSE and underperformance flags to a comparison table."""

    if comparison.empty:
        return comparison

    enriched = comparison.copy()
    naive_rmse = (
        enriched.loc[enriched["model"] == "naive", ["country", "rmse"]]
        .rename(columns={"rmse": "naive_rmse"})
        .set_index("country")["naive_rmse"]
    )
    enriched["naive_rmse"] = enriched["country"].map(naive_rmse)
    enriched["underperforms_naive"] = (
        (enriched["model"] != "naive") & (enriched["rmse"] > enriched["naive_rmse"])
    )
    return enriched.sort_values(["country", "rmse", "model"]).reset_index(drop=True)


def combine_with_baseline_comparison(
    modern_comparison: pd.DataFrame,
    baseline_path: Path,
) -> pd.DataFrame:
    """Combine existing baseline comparison rows with modern comparison rows."""

    modern = modern_comparison.copy()
    modern["model_group"] = "modern_or_reference"
    if not baseline_path.exists():
        return modern

    baseline = pd.read_csv(baseline_path)
    baseline["model_group"] = "baseline"
    return pd.concat([baseline, modern], ignore_index=True, sort=False)


def write_forecast_outputs(
    *,
    tables_dir: Path,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    modern_comparison: pd.DataFrame,
    all_model_comparison: pd.DataFrame,
    failures: pd.DataFrame,
    naive_warnings: pd.DataFrame,
) -> None:
    """Write modern forecasting tables to disk."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(tables_dir / "modern_forecast_predictions.csv", index=False)
    fold_metrics.to_csv(tables_dir / "modern_forecast_fold_metrics.csv", index=False)
    modern_comparison.to_csv(
        tables_dir / "modern_forecast_comparison.csv", index=False
    )
    all_model_comparison.to_csv(tables_dir / "all_model_comparison.csv", index=False)
    failures.to_csv(tables_dir / "modern_forecast_failed_model_runs.csv", index=False)
    naive_warnings.to_csv(
        tables_dir / "modern_forecast_naive_warnings.csv", index=False
    )


def summarize_predictions(prediction_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert prediction rows into a model comparison table."""

    predictions = pd.DataFrame(prediction_rows)
    comparison = summarize_model_metrics(predictions)
    return add_naive_comparison(comparison)


def _ensure_dependencies(model_name: str) -> None:
    """Raise a skip exception when a model's optional dependency is absent."""

    available, reason = dependency_status(model_name)
    if not available:
        raise ForecastModelSkipped(reason)


def _run_target_only_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    model_name: str,
    train_start_year: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
    forecast_fn: Callable[[pd.DataFrame, pd.Series], pd.Series],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run expanding-window validation for a univariate target model."""

    frame = (
        country_frame.loc[
            country_frame["year"] >= train_start_year,
            ["year", TARGET_INDICATOR],
        ]
        .dropna(subset=[TARGET_INDICATOR])
        .sort_values("year")
        .reset_index(drop=True)
    )
    verify_chronological_order(frame)
    splits = expanding_window_splits(
        len(frame),
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
    )
    if not splits:
        raise ForecastModelSkipped(
            f"{model_name} has insufficient target observations for {country}."
        )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        train = frame.iloc[split.train_start : split.train_end]
        test = frame.iloc[split.test_start : split.test_end]
        if train["year"].max() >= test["year"].min():
            raise ValueError(f"Train/test ordering violation for {model_name} in {country}.")
        forecast = forecast_fn(train, test["year"])
        prediction_frame = test[["year", TARGET_INDICATOR]].copy()
        prediction_frame["predicted"] = forecast.to_numpy()
        metric_rows.append(
            _metric_record(country, model_name, split, train, test, prediction_frame)
        )
        prediction_rows.extend(
            _prediction_records(country, model_name, split.fold, train, prediction_frame)
        )
    return prediction_rows, metric_rows


def _run_lagged_feature_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    model_name: str,
    train_start_year: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
    forecast_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], pd.Series],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run expanding-window validation for lagged-feature models."""

    feature_columns = lag_feature_columns()
    required_columns = ["year", TARGET_INDICATOR, *feature_columns]
    frame = (
        country_frame.loc[country_frame["year"] >= train_start_year, required_columns]
        .dropna(subset=[TARGET_INDICATOR, *feature_columns])
        .sort_values("year")
        .reset_index(drop=True)
    )
    verify_chronological_order(frame)
    splits = expanding_window_splits(
        len(frame),
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
    )
    if not splits:
        raise ForecastModelSkipped(
            f"{model_name} has insufficient complete lagged-feature rows for {country}."
        )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        train = frame.iloc[split.train_start : split.train_end]
        test = frame.iloc[split.test_start : split.test_end]
        if train["year"].max() >= test["year"].min():
            raise ValueError(f"Train/test ordering violation for {model_name} in {country}.")
        forecast = forecast_fn(
            train[feature_columns],
            train[TARGET_INDICATOR],
            test[feature_columns],
        )
        prediction_frame = test[["year", TARGET_INDICATOR]].copy()
        prediction_frame["predicted"] = forecast.to_numpy()
        metric_rows.append(
            _metric_record(country, model_name, split, train, test, prediction_frame)
        )
        prediction_rows.extend(
            _prediction_records(country, model_name, split.fold, train, prediction_frame)
        )
    return prediction_rows, metric_rows


def _metric_record(
    country: str,
    model_name: str,
    split: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Create one fold-level metric record."""

    metrics = evaluate_predictions(
        prediction_frame[TARGET_INDICATOR],
        prediction_frame["predicted"],
    )
    return {
        "country": country,
        "model": model_name,
        "fold": split.fold,
        "train_start_year": int(train["year"].min()),
        "train_end_year": int(train["year"].max()),
        "test_start_year": int(test["year"].min()),
        "test_end_year": int(test["year"].max()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        **metrics,
    }


def _prediction_records(
    country: str,
    model_name: str,
    fold: int,
    train: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create serializable prediction rows."""

    records: list[dict[str, Any]] = []
    for row in prediction_frame.itertuples(index=False):
        actual = float(getattr(row, TARGET_INDICATOR))
        predicted = float(row.predicted)
        records.append(
            {
                "country": country,
                "model": model_name,
                "fold": fold,
                "year": int(row.year),
                "actual": actual,
                "predicted": predicted,
                "residual": actual - predicted,
                "train_start_year": int(train["year"].min()),
                "train_end_year": int(train["year"].max()),
            }
        )
    return records
