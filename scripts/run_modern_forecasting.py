"""Run exploratory modern GDP growth forecasting benchmarks."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/algorithmic_nation_state_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/algorithmic_nation_state_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
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
from src.models.model_registry import ForecastModelSkipped, parse_model_selection
from src.models.modern_forecasting import (
    FEATURE_INDICATORS,
    TARGET_INDICATOR,
    add_lagged_features,
    add_naive_comparison,
    combine_with_baseline_comparison,
    load_macro_panel,
    macro_panel_to_wide,
    run_model_for_country,
    write_forecast_outputs,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run exploratory modern GDP growth forecasting benchmarks."
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
        "--models",
        default="all",
        help="Comma-separated model names, or 'all' for configured modern models.",
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
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def plot_actual_vs_predicted(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Save actual-vs-predicted plots for modern benchmark outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
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
        fig.savefig(output_dir / f"{slug}.png", dpi=150)
        plt.close(fig)


def plot_model_rankings(all_model_comparison: pd.DataFrame, output_path: Path) -> None:
    """Save a simple RMSE ranking plot across all available model rows."""

    if all_model_comparison.empty:
        return

    ranking = (
        all_model_comparison.dropna(subset=["rmse"])
        .groupby("model", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), countries=("country", "nunique"))
        .sort_values("mean_rmse")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ranking["model"], ranking["mean_rmse"])
    ax.invert_yaxis()
    ax.set_title("GDP Growth Forecast Model Rankings")
    ax.set_xlabel("Mean RMSE across available country runs")
    ax.set_ylabel("Model")
    ax.grid(True, axis="x", alpha=0.3)
    for index, row in ranking.reset_index(drop=True).iterrows():
        ax.text(
            row["mean_rmse"],
            index,
            f"  n={int(row['countries'])}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def naive_warning_rows(modern_comparison: pd.DataFrame) -> pd.DataFrame:
    """Return model/country rows that underperform the naive forecast."""

    if modern_comparison.empty or "underperforms_naive" not in modern_comparison:
        return pd.DataFrame(columns=["country", "model", "rmse", "naive_rmse", "warning"])

    warnings = modern_comparison.loc[
        modern_comparison["underperforms_naive"],
        ["country", "model", "rmse", "naive_rmse"],
    ].copy()
    warnings["warning"] = "Model RMSE is worse than the naive last-value benchmark."
    return warnings


def main() -> None:
    """Run modern forecasting benchmarks and save outputs."""

    configure_logging()
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_indicator_config(args.config)
    train_start_year = int(config.get("forecasting", {}).get("train_start_year", 2000))
    target = config.get("forecast_target", {}).get("primary", {}).get("variable", TARGET_INDICATOR)
    if target != TARGET_INDICATOR:
        raise ValueError(
            f"This modern pipeline is scoped to {TARGET_INDICATOR}, but config target is {target}."
        )

    selected_models = parse_model_selection(args.models)
    runnable_models = ["naive", *[name for name in selected_models if name != "naive"]]
    LOGGER.info("Running modern forecasting models: %s", runnable_models)

    panel = load_macro_panel(args.panel)
    wide = macro_panel_to_wide(panel, [TARGET_INDICATOR, *FEATURE_INDICATORS])
    country_codes = [country["code"] for country in config.get("countries", [])]
    if not country_codes:
        country_codes = sorted(wide["country"].dropna().unique().tolist())

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for country in country_codes:
        country_frame = wide.loc[wide["country"] == country].copy()
        if country_frame.empty:
            reason = "No country rows in processed macro panel."
            LOGGER.error("Skipping %s: %s", country, reason)
            for model_name in runnable_models:
                failure_rows.append({"country": country, "model": model_name, "reason": reason})
            continue

        country_frame = add_lagged_features(country_frame)
        missing_target_count = int(country_frame[TARGET_INDICATOR].isna().sum())
        if missing_target_count:
            LOGGER.warning(
                "%s has %s missing %s target rows.",
                country,
                missing_target_count,
                TARGET_INDICATOR,
            )

        for model_name in runnable_models:
            try:
                predictions, metrics = run_model_for_country(
                    model_name,
                    country,
                    country_frame,
                    train_start_year=train_start_year,
                    n_splits=args.n_splits,
                    test_size=args.test_size,
                )
                prediction_rows.extend(predictions)
                fold_metric_rows.extend(metrics)
                LOGGER.info("Completed %s for %s.", model_name, country)
            except ForecastModelSkipped as exc:
                reason = str(exc)
                LOGGER.warning("Skipped %s for %s: %s", model_name, country, reason)
                failure_rows.append({"country": country, "model": model_name, "reason": reason})
            except Exception as exc:  # noqa: BLE001 - keep other country/model runs alive.
                LOGGER.exception("Failed %s for %s: %s", model_name, country, exc)
                failure_rows.append({"country": country, "model": model_name, "reason": str(exc)})

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    modern_comparison = add_naive_comparison(
        pd.DataFrame() if predictions.empty else _summarize_predictions(predictions)
    )
    all_model_comparison = combine_with_baseline_comparison(
        modern_comparison,
        args.tables_dir / "forecast_model_comparison.csv",
    )
    failures = pd.DataFrame(failure_rows, columns=["country", "model", "reason"])
    warnings = naive_warning_rows(modern_comparison)

    write_forecast_outputs(
        tables_dir=args.tables_dir,
        predictions=predictions,
        fold_metrics=fold_metrics,
        modern_comparison=modern_comparison,
        all_model_comparison=all_model_comparison,
        failures=failures,
        naive_warnings=warnings,
    )

    if not predictions.empty:
        plot_actual_vs_predicted(
            predictions,
            args.figures_dir / "modern_model_actual_vs_predicted",
        )
    plot_model_rankings(
        all_model_comparison,
        args.figures_dir / "model_rankings.png",
    )

    for row in warnings.itertuples(index=False):
        LOGGER.warning(
            "%s underperforms naive for %s: model RMSE %.3f vs naive RMSE %.3f",
            row.model,
            row.country,
            row.rmse,
            row.naive_rmse,
        )

    LOGGER.info("Saved %s modern prediction rows.", len(predictions))
    LOGGER.info("Saved %s modern comparison rows.", len(modern_comparison))
    if not failures.empty:
        LOGGER.warning("Some models were skipped or failed; see modern_forecast_failed_model_runs.csv")


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction rows without importing plotting code elsewhere."""

    from src.models.evaluation import summarize_model_metrics

    return summarize_model_metrics(predictions)


if __name__ == "__main__":
    main()
