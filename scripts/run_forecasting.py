"""Run baseline GDP growth forecasting models."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/algorithmic_nation_state_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/algorithmic_nation_state_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_indicator_config
from src.models.arima_model import forecast_arima
from src.models.evaluation import (
    evaluate_predictions,
    expanding_window_splits,
    summarize_model_metrics,
    verify_chronological_order,
)
from src.models.random_forest_model import forecast_random_forest
from src.models.sarimax_model import forecast_sarimax

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run baseline GDP growth forecasting models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/indicators.yaml",
        help="Path to indicator YAML configuration.",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=PROJECT_ROOT / "data/processed/macro_panel.csv",
        help="Path to cleaned macro panel CSV.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/tables",
        help="Directory for forecast tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/figures",
        help="Directory for forecast figures.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Maximum expanding-window validation folds.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=3,
        help="Years per validation test fold.",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=8,
        help="Minimum training years required for a validation fold.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_macro_panel(panel_path: Path) -> pd.DataFrame:
    """Load and validate the long macro panel produced by data ingestion."""

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
    """Pivot the long panel to one country-year row per observation."""

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
    target: str,
    features: list[str],
) -> pd.DataFrame:
    """Add one-year lags to avoid same-year feature leakage."""

    lagged = country_frame.sort_values("year").copy()
    for column in [target, *features]:
        if column in lagged.columns:
            # Assumption: year t GDP growth is predicted only from information
            # observed by year t-1, so features are shifted one year backward.
            lagged[f"{column}_L1"] = lagged[column].shift(1)
    return lagged


def run_arima_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    train_start_year: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run expanding-window ARIMA validation for one country."""

    frame = (
        country_frame.loc[country_frame["year"] >= train_start_year, ["year", TARGET_INDICATOR]]
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
        raise ValueError(f"Not enough target observations for ARIMA in {country}.")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        train = frame.iloc[split.train_start : split.train_end]
        test = frame.iloc[split.test_start : split.test_end]
        if train["year"].max() >= test["year"].min():
            raise ValueError(f"Train/test ordering violation for ARIMA in {country}.")

        forecast = forecast_arima(train[TARGET_INDICATOR], steps=len(test))
        fold_predictions = test[["year", TARGET_INDICATOR]].copy()
        fold_predictions["predicted"] = forecast.to_numpy()
        metrics = evaluate_predictions(
            fold_predictions[TARGET_INDICATOR],
            fold_predictions["predicted"],
        )
        metric_rows.append(
            {
                "country": country,
                "model": "ARIMA",
                "fold": split.fold,
                "train_start_year": int(train["year"].min()),
                "train_end_year": int(train["year"].max()),
                "test_start_year": int(test["year"].min()),
                "test_end_year": int(test["year"].max()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
            }
        )
        prediction_rows.extend(
            _prediction_records(country, "ARIMA", split.fold, train, fold_predictions)
        )
    return prediction_rows, metric_rows


def run_sarimax_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    feature_lags: list[str],
    train_start_year: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run expanding-window SARIMAX validation for one country."""

    required_columns = ["year", TARGET_INDICATOR, *feature_lags]
    frame = (
        country_frame.loc[country_frame["year"] >= train_start_year, required_columns]
        .dropna(subset=[TARGET_INDICATOR, *feature_lags])
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
        raise ValueError(f"Not enough complete lagged feature rows for SARIMAX in {country}.")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        train = frame.iloc[split.train_start : split.train_end]
        test = frame.iloc[split.test_start : split.test_end]
        if train["year"].max() >= test["year"].min():
            raise ValueError(f"Train/test ordering violation for SARIMAX in {country}.")

        forecast = forecast_sarimax(
            train[TARGET_INDICATOR],
            train[feature_lags],
            test[feature_lags],
        )
        fold_predictions = test[["year", TARGET_INDICATOR]].copy()
        fold_predictions["predicted"] = forecast.to_numpy()
        metrics = evaluate_predictions(
            fold_predictions[TARGET_INDICATOR],
            fold_predictions["predicted"],
        )
        metric_rows.append(
            {
                "country": country,
                "model": "SARIMAX",
                "fold": split.fold,
                "train_start_year": int(train["year"].min()),
                "train_end_year": int(train["year"].max()),
                "test_start_year": int(test["year"].min()),
                "test_end_year": int(test["year"].max()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
            }
        )
        prediction_rows.extend(
            _prediction_records(country, "SARIMAX", split.fold, train, fold_predictions)
        )
    return prediction_rows, metric_rows


def run_random_forest_country(
    country: str,
    country_frame: pd.DataFrame,
    *,
    feature_lags: list[str],
    train_start_year: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run expanding-window RandomForest validation for one country."""

    rf_features = [f"{TARGET_INDICATOR}_L1", *feature_lags]
    required_columns = ["year", TARGET_INDICATOR, *rf_features]
    frame = (
        country_frame.loc[country_frame["year"] >= train_start_year, required_columns]
        .dropna(subset=[TARGET_INDICATOR, *rf_features])
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
        raise ValueError(
            f"Not enough complete lagged feature rows for RandomForest in {country}."
        )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for split in splits:
        train = frame.iloc[split.train_start : split.train_end]
        test = frame.iloc[split.test_start : split.test_end]
        if train["year"].max() >= test["year"].min():
            raise ValueError(
                f"Train/test ordering violation for RandomForest in {country}."
            )

        forecast = forecast_random_forest(
            train[rf_features],
            train[TARGET_INDICATOR],
            test[rf_features],
        )
        fold_predictions = test[["year", TARGET_INDICATOR]].copy()
        fold_predictions["predicted"] = forecast.to_numpy()
        metrics = evaluate_predictions(
            fold_predictions[TARGET_INDICATOR],
            fold_predictions["predicted"],
        )
        metric_rows.append(
            {
                "country": country,
                "model": "RandomForest",
                "fold": split.fold,
                "train_start_year": int(train["year"].min()),
                "train_end_year": int(train["year"].max()),
                "test_start_year": int(test["year"].min()),
                "test_end_year": int(test["year"].max()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
            }
        )
        prediction_rows.extend(
            _prediction_records(
                country,
                "RandomForest",
                split.fold,
                train,
                fold_predictions,
            )
        )
    return prediction_rows, metric_rows


def _prediction_records(
    country: str,
    model: str,
    fold: int,
    train: pd.DataFrame,
    fold_predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert fold predictions to serializable rows."""

    records: list[dict[str, Any]] = []
    for row in fold_predictions.itertuples(index=False):
        actual = float(getattr(row, TARGET_INDICATOR))
        predicted = float(row.predicted)
        records.append(
            {
                "country": country,
                "model": model,
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


def plot_forecasts(predictions: pd.DataFrame, figures_dir: Path) -> None:
    """Create actual-vs-predicted and residual plots for each model/country."""

    for (country, model), group in predictions.groupby(["country", "model"]):
        group = group.sort_values("year")
        slug = f"{country.lower()}_{model.lower()}"

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(group["year"], group["actual"], marker="o", label="Actual")
        ax.plot(group["year"], group["predicted"], marker="o", label="Predicted")
        ax.set_title(f"{country} GDP Growth: {model}")
        ax.set_xlabel("Year")
        ax.set_ylabel("GDP growth (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / f"forecast_actual_vs_predicted_{slug}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(group["year"], group["residual"])
        ax.set_title(f"{country} Forecast Residuals: {model}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Actual - predicted")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / f"forecast_residuals_{slug}.png", dpi=150)
        plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, figures_dir: Path) -> None:
    """Create a country-level RMSE comparison plot."""

    if comparison.empty:
        return

    pivot = comparison.pivot(index="country", columns="model", values="rmse").sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("GDP Growth Forecast Baseline Comparison")
    ax.set_xlabel("Country")
    ax.set_ylabel("RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "forecast_model_comparison_rmse.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run all baseline forecasts and save tables and figures."""

    configure_logging()
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_indicator_config(args.config)
    train_start_year = int(config.get("forecasting", {}).get("train_start_year", 2000))
    target = config.get("forecast_target", {}).get("primary", {}).get("variable", TARGET_INDICATOR)
    if target != TARGET_INDICATOR:
        raise ValueError(
            f"This baseline pipeline is scoped to {TARGET_INDICATOR}, but config target is {target}."
        )

    panel = load_macro_panel(args.panel)
    indicators = [TARGET_INDICATOR, *FEATURE_INDICATORS]
    wide = macro_panel_to_wide(panel, indicators)
    feature_lags = [f"{feature}_L1" for feature in FEATURE_INDICATORS]

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    country_codes = [country["code"] for country in config.get("countries", [])]
    if not country_codes:
        country_codes = sorted(wide["country"].dropna().unique().tolist())

    for country in country_codes:
        country_frame = wide.loc[wide["country"] == country].copy()
        if country_frame.empty:
            failed_rows.append(
                {
                    "country": country,
                    "model": "ALL",
                    "reason": "No country rows in processed macro panel.",
                }
            )
            LOGGER.error("Skipping %s: no rows in processed macro panel.", country)
            continue

        country_frame = add_lagged_features(
            country_frame,
            target=TARGET_INDICATOR,
            features=FEATURE_INDICATORS,
        )
        target_missing = country_frame[TARGET_INDICATOR].isna().sum()
        if target_missing:
            LOGGER.warning(
                "%s has %s missing %s target rows.",
                country,
                int(target_missing),
                TARGET_INDICATOR,
            )
        if country_frame[TARGET_INDICATOR].notna().sum() == 0:
            failed_rows.append(
                {
                    "country": country,
                    "model": "ALL",
                    "reason": f"No non-missing {TARGET_INDICATOR} target values.",
                }
            )
            LOGGER.error("Skipping %s: no non-missing target values.", country)
            continue

        model_runners = [
            ("ARIMA", run_arima_country),
            ("SARIMAX", run_sarimax_country),
            ("RandomForest", run_random_forest_country),
        ]
        for model_name, runner in model_runners:
            try:
                if model_name == "ARIMA":
                    preds, metrics = runner(
                        country,
                        country_frame,
                        train_start_year=train_start_year,
                        n_splits=args.n_splits,
                        test_size=args.test_size,
                        min_train_size=args.min_train_size,
                    )
                else:
                    preds, metrics = runner(
                        country,
                        country_frame,
                        feature_lags=feature_lags,
                        train_start_year=train_start_year,
                        n_splits=args.n_splits,
                        test_size=args.test_size,
                        min_train_size=args.min_train_size,
                    )
                prediction_rows.extend(preds)
                fold_metric_rows.extend(metrics)
                LOGGER.info("Completed %s baseline for %s.", model_name, country)
            except Exception as exc:  # noqa: BLE001 - keep the multi-country run alive.
                LOGGER.exception("Failed %s baseline for %s: %s", model_name, country, exc)
                failed_rows.append(
                    {
                        "country": country,
                        "model": model_name,
                        "reason": str(exc),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    comparison = summarize_model_metrics(predictions)
    failures = pd.DataFrame(failed_rows, columns=["country", "model", "reason"])

    predictions.to_csv(args.tables_dir / "forecast_predictions.csv", index=False)
    fold_metrics.to_csv(args.tables_dir / "forecast_fold_metrics.csv", index=False)
    comparison.to_csv(args.tables_dir / "forecast_model_comparison.csv", index=False)
    failures.to_csv(args.tables_dir / "forecast_failed_country_runs.csv", index=False)

    if not predictions.empty:
        plot_forecasts(predictions, args.figures_dir)
        plot_model_comparison(comparison, args.figures_dir)

    LOGGER.info("Saved %s forecast prediction rows.", len(predictions))
    LOGGER.info("Saved %s model comparison rows.", len(comparison))
    if not failures.empty:
        LOGGER.warning("Some country/model runs failed; see forecast_failed_country_runs.csv")


if __name__ == "__main__":
    main()
