"""Run interpretability analysis for tree-based GDP forecasting models."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/algorithmic_nation_state_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/algorithmic_nation_state_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_indicator_config
from src.models.interpretability import (
    TARGET_INDICATOR,
    complete_model_frame,
    compute_correlation_outputs,
    compute_importance_for_frame,
    compute_lag_relationships,
    data_quality_summary,
    prepare_interpretability_dataset,
    rolling_importance,
    summarize_global_importance,
)
from src.visualization.importance_plots import (
    plot_correlation_heatmap,
    plot_country_feature_importance,
    plot_global_feature_importance,
    plot_shap_summary,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run tree-model interpretability for GDP growth forecasts."
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
        help="Directory for interpretability tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/figures",
        help="Directory for interpretability figures.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/interpretability_summary.md",
        help="Markdown interpretability report path.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=3,
        help="Chronological holdout years for permutation importance.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    """Run interpretability workflow and write tables, figures, and report."""

    configure_logging()
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_indicator_config(args.config)
    train_start_year = int(config.get("forecasting", {}).get("train_start_year", 2000))
    countries = [country["code"] for country in config.get("countries", [])]

    dataset = prepare_interpretability_dataset(args.panel)
    frame = complete_model_frame(
        dataset.wide,
        dataset.feature_columns,
        train_start_year=train_start_year,
    )
    quality = data_quality_summary(frame, countries)
    quality.to_csv(args.tables_dir / "interpretability_data_quality.csv", index=False)

    global_frame = complete_model_frame(
        dataset.wide,
        dataset.feature_columns,
        train_start_year=train_start_year,
    )
    global_importance_raw, global_shap, global_failures = compute_importance_for_frame(
        frame=global_frame,
        feature_columns=dataset.feature_columns,
        scope="global",
        country="GLOBAL",
        train_start_year=train_start_year,
        test_size=args.test_size,
        min_train_rows=24,
        country_level=False,
    )

    country_importance_rows: list[pd.DataFrame] = []
    shap_rows: list[pd.DataFrame] = [global_shap] if not global_shap.empty else []
    failure_rows = list(global_failures)
    for country in countries:
        country_frame = complete_model_frame(
            dataset.wide,
            dataset.feature_columns,
            country=country,
            train_start_year=train_start_year,
        )
        importance, shap_summary, failures = compute_importance_for_frame(
            frame=country_frame,
            feature_columns=dataset.feature_columns,
            scope="country",
            country=country,
            train_start_year=train_start_year,
            test_size=args.test_size,
            min_train_rows=8,
            country_level=True,
        )
        if not importance.empty:
            country_importance_rows.append(importance)
        if not shap_summary.empty:
            shap_rows.append(shap_summary)
        failure_rows.extend(failures)

    by_country = (
        pd.concat(country_importance_rows, ignore_index=True)
        if country_importance_rows
        else pd.DataFrame()
    )
    global_summary = summarize_global_importance(by_country)
    all_importance = pd.concat(
        [frame for frame in [global_importance_raw, by_country] if not frame.empty],
        ignore_index=True,
    )
    shap_summary = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    failures = pd.DataFrame(failure_rows, columns=["scope", "country", "model", "reason"])
    correlation_matrix, correlation_summary = compute_correlation_outputs(
        frame,
        dataset.feature_columns,
    )
    lag_relationships = compute_lag_relationships(frame, dataset.feature_columns)
    rolling = rolling_importance(frame, dataset.feature_columns)

    global_summary.to_csv(args.tables_dir / "feature_importance_global.csv", index=False)
    by_country.to_csv(args.tables_dir / "feature_importance_by_country.csv", index=False)
    all_importance.to_csv(args.tables_dir / "feature_importance_all_scopes.csv", index=False)
    correlation_matrix.to_csv(args.tables_dir / "correlation_matrix.csv")
    correlation_summary.to_csv(args.tables_dir / "correlation_summary.csv", index=False)
    lag_relationships.to_csv(args.tables_dir / "lag_relationships.csv", index=False)
    rolling.to_csv(args.tables_dir / "rolling_feature_importance.csv", index=False)
    shap_summary.to_csv(args.tables_dir / "shap_summary.csv", index=False)
    failures.to_csv(args.tables_dir / "interpretability_failed_runs.csv", index=False)

    plot_global_feature_importance(
        global_summary,
        args.figures_dir / "global_feature_importance.png",
    )
    plot_country_feature_importance(
        by_country,
        args.figures_dir / "country_feature_importance",
    )
    plot_shap_summary(
        shap_summary,
        args.figures_dir / "shap_summary",
    )
    plot_correlation_heatmap(
        correlation_matrix,
        args.figures_dir / "correlation_heatmap.png",
    )

    write_markdown_report(
        report_path=args.report,
        global_summary=global_summary,
        by_country=by_country,
        correlation_summary=correlation_summary,
        quality=quality,
        failures=failures,
    )

    log_weak_and_unstable(global_summary, by_country)
    LOGGER.info("Wrote interpretability report to %s", args.report)
    LOGGER.info("Feature importance rows by country: %s", len(by_country))


def write_markdown_report(
    *,
    report_path: Path,
    global_summary: pd.DataFrame,
    by_country: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    quality: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    """Write a conservative markdown summary of interpretability results."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    top_global = global_summary.head(8)
    stable = global_summary.loc[global_summary["stability_warning"] == ""].head(5)
    unstable = global_summary.loc[global_summary["stability_warning"] != ""].head(8)

    lines = [
        "# Interpretability Summary",
        "",
        "This report summarizes exploratory feature-importance analysis for GDP growth forecasting. "
        "The analysis uses one-year-lagged macroeconomic indicators only; contemporaneous target values are not used as predictors.",
        "",
        "## Data Coverage",
        "",
        markdown_table(quality),
        "",
        "## Strongest Predictive Indicators",
        "",
    ]
    if top_global.empty:
        lines.append("No feature-importance results were generated.")
    else:
        lines.append(
            markdown_table(
                top_global[
                    [
                        "model",
                        "feature_base",
                        "feature_group",
                        "mean_permutation_importance_mean",
                        "mean_native_importance",
                        "countries",
                    ]
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Stable Signals",
            "",
            markdown_table(
                stable[
                    [
                        "model",
                        "feature_base",
                        "feature_group",
                        "mean_permutation_importance_mean",
                        "countries",
                    ]
                ]
            )
            if not stable.empty
            else "No clearly stable importance signals were identified.",
            "",
            "## Unstable Or Weak Findings",
            "",
            markdown_table(
                unstable[
                    [
                        "model",
                        "feature_base",
                        "feature_group",
                        "mean_permutation_importance_mean",
                        "instability_score",
                        "stability_warning",
                    ]
                ]
            )
            if not unstable.empty
            else "Most computed global importance rows were not flagged by the simple instability heuristic.",
            "",
            "## Correlation Notes",
            "",
        ]
    )
    corr_global = correlation_summary.loc[correlation_summary["country"] == "GLOBAL"].copy()
    corr_global["abs_corr"] = corr_global["correlation_with_target"].abs()
    if not corr_global.empty:
        lines.append(
            markdown_table(
                corr_global.sort_values("abs_corr", ascending=False)
                .head(8)[
                    [
                        "feature_base",
                        "feature_group",
                        "correlation_with_target",
                        "n_rows",
                    ]
                ]
            )
        )
    else:
        lines.append("No correlation summary was available.")

    lines.extend(
        [
            "",
            "## Methodological Caveats",
            "",
            "- Feature importance is predictive, not causal.",
            "- Correlation is not causation and should not be interpreted as policy evidence by itself.",
            "- The dataset is annual and small, especially after one-year lagging and complete-case filtering.",
            "- Permutation importance is measured on small chronological holdout sets and can be noisy.",
            "- SHAP-style outputs use LightGBM/XGBoost native tree contribution methods where available; RandomForest SHAP was skipped because the `shap` package is not installed.",
            "- Taiwan has no complete World Bank panel rows in the current processed dataset.",
            "",
            "## Failed Or Skipped Interpretability Runs",
            "",
            markdown_table(failures) if not failures.empty else "No failures were recorded.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table."""

    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def log_weak_and_unstable(global_summary: pd.DataFrame, by_country: pd.DataFrame) -> None:
    """Log weak and unstable feature-importance diagnostics."""

    if global_summary.empty:
        LOGGER.warning("No global feature-importance rows were generated.")
        return
    unstable = global_summary.loc[global_summary["stability_warning"] != ""]
    if not unstable.empty:
        LOGGER.warning(
            "Unstable global importance rows detected: %s",
            unstable[["model", "feature_base", "stability_warning"]].head(10).to_dict("records"),
        )
    weak = by_country.loc[
        by_country.get("importance_warning", pd.Series(dtype=str)).astype(str) != ""
    ]
    if not weak.empty:
        LOGGER.warning(
            "Weak country-level importance rows detected: %s",
            weak[["country", "model", "feature_base", "importance_warning"]]
            .head(10)
            .to_dict("records"),
        )


if __name__ == "__main__":
    main()
