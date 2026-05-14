"""Generate integrated report assets from existing analysis outputs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/algorithmic_nation_state_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/algorithmic_nation_state_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import build_project_paths, ensure_project_directories, load_yaml_config
from src.utils.export_utils import markdown_table, read_csv_if_exists, table_summary, write_markdown
from src.utils.logging_utils import configure_project_logging
from src.utils.reproducibility import collect_environment_metadata, write_metadata

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate integrated report assets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/indicators.yaml",
        help="Path to project config.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate integrated figures and markdown reports."""

    args = parse_args()
    paths = build_project_paths(project_root=args.project_root, config_path=args.config)
    ensure_project_directories(paths)
    logger, log_path = configure_project_logging(logs_dir=paths.logs, name="report_assets")
    logger.info("Writing report asset log to %s", log_path)

    config = load_yaml_config(paths.config)
    tables = load_tables(paths)
    figures_dir = paths.figures / "integrated"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_forecast_comparison(tables["all_model_comparison"], figures_dir)
    plot_feature_importance(tables["feature_importance_global"], figures_dir)
    plot_abm_trajectories(tables["abm_timeseries"], figures_dir)
    plot_bayesian_posteriors(tables["bayesian_posteriors"], figures_dir)
    plot_monte_carlo_uncertainty(tables["monte_carlo_timeseries"], figures_dir)
    plot_integrated_dashboard(tables, figures_dir)

    write_integrated_reports(paths, config, tables, figures_dir)
    write_asset_inventory(paths, figures_dir)
    metadata = collect_environment_metadata(paths.root)
    metadata["report_assets_log"] = str(log_path)
    write_metadata(paths.reports / "report_asset_metadata.json", metadata)
    logger.info("Integrated report assets generated.")


def load_tables(paths) -> dict[str, pd.DataFrame]:
    """Load all report tables used by integrated assets."""

    return {
        "macro_panel": read_csv_if_exists(paths.macro_panel),
        "forecast_model_comparison": read_csv_if_exists(paths.tables / "forecast_model_comparison.csv"),
        "all_model_comparison": read_csv_if_exists(paths.tables / "all_model_comparison.csv"),
        "feature_importance_global": read_csv_if_exists(paths.tables / "feature_importance_global.csv"),
        "correlation_summary": read_csv_if_exists(paths.tables / "correlation_summary.csv"),
        "interpretability_data_quality": read_csv_if_exists(paths.tables / "interpretability_data_quality.csv"),
        "abm_timeseries": read_csv_if_exists(paths.tables / "abm/abm_timeseries.csv"),
        "abm_scenario_comparison": read_csv_if_exists(paths.tables / "abm/abm_scenario_comparison.csv"),
        "abm_validation": read_csv_if_exists(paths.tables / "abm/abm_validation_report.csv"),
        "bayesian_posteriors": read_csv_if_exists(paths.tables / "probabilistic/bayesian_posteriors.csv"),
        "bayesian_diagnostics": read_csv_if_exists(paths.tables / "probabilistic/bayesian_diagnostics.csv"),
        "scenario_probabilities": read_csv_if_exists(paths.tables / "probabilistic/scenario_probability_updates.csv"),
        "monte_carlo_summary": read_csv_if_exists(paths.tables / "probabilistic/monte_carlo_summary.csv"),
        "monte_carlo_timeseries": read_csv_if_exists(paths.tables / "probabilistic/monte_carlo_timeseries.csv"),
        "monte_carlo_diagnostics": read_csv_if_exists(paths.tables / "probabilistic/monte_carlo_diagnostics.csv"),
        "sensitivity_rankings": read_csv_if_exists(paths.tables / "probabilistic/sensitivity_rankings.csv"),
    }


def plot_forecast_comparison(comparison: pd.DataFrame, figures_dir: Path) -> None:
    """Create a publication-style forecast RMSE comparison figure."""

    if comparison.empty or not {"model", "rmse"}.issubset(comparison.columns):
        return
    ranking = (
        comparison.dropna(subset=["rmse"])
        .groupby("model", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), countries=("country", "nunique"))
        .sort_values("mean_rmse")
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(ranking["model"], ranking["mean_rmse"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_title("GDP Growth Forecast Benchmarks")
    ax.set_xlabel("Mean RMSE across available country runs")
    ax.set_ylabel("Model")
    ax.grid(True, axis="x", alpha=0.25)
    for index, row in ranking.reset_index(drop=True).iterrows():
        ax.text(row["mean_rmse"], index, f"  n={int(row['countries'])}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "publication_forecasting_comparison.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(global_importance: pd.DataFrame, figures_dir: Path) -> None:
    """Create a publication-style feature importance figure."""

    value_column = "mean_permutation_importance_mean"
    if global_importance.empty or value_column not in global_importance:
        return
    frame = global_importance.dropna(subset=[value_column]).head(10).sort_values(value_column)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = frame["model"].astype(str) + " | " + frame["feature_base"].astype(str)
    ax.barh(labels, frame[value_column], color="#54A24B")
    ax.set_title("Exploratory Forecast Feature Importance")
    ax.set_xlabel("Mean permutation importance")
    ax.set_ylabel("Model and feature")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "publication_feature_importance.png", dpi=180)
    plt.close(fig)


def plot_abm_trajectories(timeseries: pd.DataFrame, figures_dir: Path) -> None:
    """Create ABM scenario trajectory figure."""

    if timeseries.empty:
        return
    metrics = ["economic_output_proxy", "dependency_score", "ai_capability_divergence"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric in zip(axes, metrics):
        if metric not in timeseries:
            continue
        for scenario, frame in timeseries.groupby("scenario"):
            ax.plot(frame["timestep"], frame[metric], label=scenario)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("ECAIF ABM Scenario Trajectories", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "publication_abm_scenario_trajectories.png", dpi=180)
    plt.close(fig)


def plot_bayesian_posteriors(posteriors: pd.DataFrame, figures_dir: Path) -> None:
    """Create a compact posterior mean figure."""

    required = {"event_name", "scope_type", "scope_value", "posterior_mean"}
    if posteriors.empty or not required.issubset(posteriors.columns):
        return
    frame = posteriors.loc[posteriors["scope_type"] == "scenario"].copy()
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="event_name",
        columns="scope_value",
        values="posterior_mean",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(9, 5.2))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=[item.replace("_", " ") for item in pivot.index])
    ax.set_title("Bayesian Posterior Means From Simulation Evidence Proxies")
    fig.colorbar(image, ax=ax, label="Posterior mean")
    fig.tight_layout()
    fig.savefig(figures_dir / "publication_bayesian_posteriors.png", dpi=180)
    plt.close(fig)


def plot_monte_carlo_uncertainty(timeseries: pd.DataFrame, figures_dir: Path) -> None:
    """Create Monte Carlo uncertainty-band figure."""

    metric = "economic_output_proxy"
    if timeseries.empty or metric not in timeseries:
        return
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for scenario, frame in timeseries.groupby("scenario"):
        quantiles = frame.groupby("timestep")[metric].quantile([0.05, 0.50, 0.95]).unstack()
        x = quantiles.index.to_numpy(dtype=float)
        lower = quantiles[0.05].to_numpy(dtype=float)
        median = quantiles[0.50].to_numpy(dtype=float)
        upper = quantiles[0.95].to_numpy(dtype=float)
        ax.plot(x, median, label=scenario)
        ax.fill_between(x, lower, upper, alpha=0.12)
    ax.set_title("Monte Carlo Uncertainty Bands")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Economic output proxy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "publication_monte_carlo_uncertainty_bands.png", dpi=180)
    plt.close(fig)


def plot_integrated_dashboard(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    """Create a compact integrated dashboard from core outputs."""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _dashboard_forecasts(axes[0, 0], tables["all_model_comparison"])
    _dashboard_features(axes[0, 1], tables["feature_importance_global"])
    _dashboard_abm(axes[0, 2], tables["abm_scenario_comparison"])
    _dashboard_bayes(axes[1, 0], tables["scenario_probabilities"])
    _dashboard_monte_carlo(axes[1, 1], tables["monte_carlo_summary"])
    _dashboard_data_coverage(axes[1, 2], tables["macro_panel"])
    fig.suptitle("Algorithmic Nation-State Integrated Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "integrated_summary_dashboard.png", dpi=180)
    plt.close(fig)


def write_integrated_reports(
    paths,
    config: dict,
    tables: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Write integrated markdown reports."""

    write_markdown(
        paths.reports / "main_analysis_results.md",
        main_results_markdown(config, tables, figures_dir),
    )
    write_markdown(paths.reports / "methodology_summary.md", methodology_markdown())
    write_markdown(paths.reports / "limitations_and_assumptions.md", limitations_markdown())
    write_markdown(paths.reports / "project_architecture.md", architecture_markdown())


def main_results_markdown(config: dict, tables: dict[str, pd.DataFrame], figures_dir: Path) -> str:
    """Build conservative integrated results markdown."""

    macro = tables["macro_panel"]
    coverage = pd.DataFrame()
    if not macro.empty:
        configured_countries = [
            country.get("code", "") for country in config.get("countries", [])
        ]
        coverage = (
            macro.groupby("country", as_index=False)
            .agg(rows=("value", "size"), observed_values=("value", lambda s: int(s.notna().sum())))
            .set_index("country")
            .reindex(configured_countries)
            .fillna(0)
            .astype({"rows": int, "observed_values": int})
            .reset_index()
            .sort_values("country")
        )
    forecast = _forecast_summary(tables["all_model_comparison"])
    feature = _feature_summary(tables["feature_importance_global"])
    abm = _abm_summary(tables["abm_scenario_comparison"])
    bayes = _bayes_summary(tables["bayesian_posteriors"])
    mc = _mc_summary(tables["monte_carlo_summary"])
    sensitivity = _sensitivity_summary(tables["sensitivity_rankings"])
    target = config.get("forecast_target", {}).get("primary", {}).get("variable", "GDP_GROWTH")
    return f"""# Main Analysis Results

This report integrates the current reproducible outputs for the Algorithmic
Nation-State project. Findings are intentionally conservative: empirical
forecasting diagnostics, exploratory simulations, and subjective probabilistic
updates are separated rather than blended into a single prediction.

## Scope

- Forecast target: `{target}`
- Countries: `{", ".join(country.get("code", "") for country in config.get("countries", []))}`
- Integrated figures: `{figures_dir}`

## Empirical Data Coverage

{markdown_table(coverage, max_rows=12)}

## Forecasting Benchmarks

{markdown_table(forecast, max_rows=10)}

These rows summarize observed model errors from the generated comparison table.
They should not be read as evidence of structural causal relationships.

## Forecast Interpretability

{markdown_table(feature, max_rows=10)}

Feature importance is predictive and model-dependent. It is not causal evidence.

## ECAIF ABM Scenario Outputs

{markdown_table(abm, max_rows=10)}

ABM values are simulated diagnostics conditional on stated assumptions.

## Bayesian Update Outputs

{markdown_table(bayes, max_rows=12)}

Posterior values use weak priors and simulation/model evidence proxies. They are
not calibrated real-world geopolitical probabilities.

## Monte Carlo Summary

{markdown_table(mc, max_rows=12)}

## Major Sensitivity Drivers

{markdown_table(sensitivity, max_rows=10)}

Sensitivity rankings are rank correlations over sampled ABM perturbations, not
causal decompositions.
"""


def methodology_markdown() -> str:
    """Return a compact methodology summary."""

    return """# Methodology Summary

## Data

World Bank indicators are loaded from `config/indicators.yaml`, stored as raw
JSON, and cleaned into `data/processed/macro_panel.csv`. Missing values are
reported and are not silently filled.

## Forecasting

GDP growth forecasting uses chronological validation. ARIMA is univariate;
SARIMAX, RandomForest, LightGBM, and other optional benchmarks use lagged
features where appropriate to avoid same-year leakage.

## Interpretability

Feature importance uses tree-model permutation importance, native importance,
correlations, lag diagnostics, and contribution summaries where practical.

## Agent-Based Modeling

The ECAIF ABM models states, AI firms, and compute providers with stylized
resources. Scenarios alter openness, cooperation, shocks, compute growth, data
sharing, capital mobility, and R&D reinvestment.

## Probabilistic Layer

Bayesian updates use closed-form Beta and Dirichlet structures over explicitly
labeled evidence proxies. Monte Carlo analysis samples transparent scenario
perturbations and reruns the ABM.

## Reproducibility

Scripts centralize configuration loading, logging, run metadata, deterministic
seeds, schema checks, and output locations.
"""


def limitations_markdown() -> str:
    """Return assumptions and limitations markdown."""

    return """# Limitations And Assumptions

## Empirical Limits

- The macroeconomic dataset is annual and small after country-level filtering.
- OECD ingestion is not yet implemented.
- Taiwan lacks World Bank panel observations in the current processed dataset.
- Forecasting benchmarks are baseline diagnostics, not validated predictive
  systems.

## Simulation Limits

- ABM resources are stylized indexes, not measured physical stocks.
- Shocks are scenario assumptions, not empirical event probabilities.
- Alliance and market-access rules are conceptual simplifications.
- Economic output proxy is not GDP.

## Probabilistic Limits

- Bayesian priors are weak subjective priors unless future calibration files are
  supplied.
- Scenario probability weights are stress-evidence weights, not forecasts of
  geopolitical futures.
- Monte Carlo distributions are inspectable stress-test ranges, not calibrated
  empirical distributions.

## Interpretation Rules

- Do not make causal claims from feature importance, correlations, or
  sensitivity rankings.
- Do not present simulated outputs as geopolitical predictions.
- Do not hide missingness or calibration gaps.
"""


def architecture_markdown() -> str:
    """Return project architecture documentation."""

    return """# Project Architecture

## Module Overview

- `src/data`: World Bank ingestion, raw response storage, panel cleaning, and
  missingness reporting.
- `src/models`: baseline forecasting, modern benchmarks, evaluation, and
  interpretability.
- `src/abm`: ECAIF agents, resources, shocks, environment, metrics, and
  scenario simulation.
- `src/bayesian`: weak priors, evidence proxies, posterior updates, and scenario
  probability evolution.
- `src/monte_carlo`: parameter distributions, scenario perturbation, repeated
  ABM runs, and sensitivity rankings.
- `src/utils`: configuration, logging, reproducibility, and export helpers.

## Data Flow

```mermaid
flowchart LR
  A["config/indicators.yaml"] --> B["World Bank ingestion"]
  B --> C["data/processed/macro_panel.csv"]
  C --> D["Forecasting"]
  C --> E["Interpretability"]
  C --> F["ABM initialization"]
  F --> G["ABM scenario outputs"]
  D --> H["Bayesian evidence"]
  G --> H
  G --> I["Monte Carlo propagation"]
  H --> J["Integrated reports"]
  I --> J
  E --> J
```

## Model Relationships

Forecasting produces GDP-growth benchmark errors and predictions.
Interpretability explains tree-model signal structure using lagged predictors.
ABM outputs provide scenario dynamics for conceptual mechanisms. Bayesian and
Monte Carlo layers quantify uncertainty conditional on those assumptions.

## Reproducibility Strategy

All scripts resolve paths from the shared project config, write logs under
`reports/logs/`, record metadata under `reports/`, use deterministic seeds where
applicable, and preserve generated tables and figures under the reports tree.
"""


def write_asset_inventory(paths, figures_dir: Path) -> None:
    """Write a CSV inventory of report tables and integrated figures."""

    table_paths = sorted(paths.tables.rglob("*.csv"))
    figure_paths = sorted(figures_dir.glob("*.png"))
    rows = [table_summary(path) for path in table_paths]
    rows.extend(
        {"path": str(path), "exists": path.exists(), "rows": 0, "columns": 0}
        for path in figure_paths
    )
    pd.DataFrame(rows).to_csv(paths.tables / "report_asset_inventory.csv", index=False)


def _forecast_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty or not {"model", "rmse"}.issubset(comparison.columns):
        return pd.DataFrame()
    return (
        comparison.dropna(subset=["rmse"])
        .groupby("model", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), countries=("country", "nunique"))
        .sort_values("mean_rmse")
    )


def _feature_summary(global_importance: pd.DataFrame) -> pd.DataFrame:
    value = "mean_permutation_importance_mean"
    columns = ["model", "feature_base", "feature_group", value, "countries", "stability_warning"]
    if global_importance.empty or value not in global_importance:
        return pd.DataFrame()
    return global_importance[[column for column in columns if column in global_importance]].head(10)


def _abm_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "final_economic_output_proxy",
        "final_dependency_score",
        "final_ai_capability_divergence",
        "total_shock_events",
    ]
    if comparison.empty:
        return pd.DataFrame()
    return comparison[[column for column in columns if column in comparison]]


def _bayes_summary(posteriors: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_name",
        "scope_type",
        "scope_value",
        "posterior_mean",
        "credible_interval_lower",
        "credible_interval_upper",
    ]
    if posteriors.empty:
        return pd.DataFrame()
    return posteriors[[column for column in columns if column in posteriors]].head(12)


def _mc_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    frame = summary.loc[
        summary["metric"].isin(
            [
                "final_economic_output_proxy",
                "final_dependency_score",
                "final_ai_capability_divergence",
            ]
        )
    ]
    return frame[["scenario", "metric", "mean", "q05", "q95", "unstable_run_rate"]].head(12)


def _sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    if sensitivity.empty:
        return pd.DataFrame()
    return (
        sensitivity.dropna(subset=["absolute_correlation"])
        .groupby("parameter", as_index=False)
        .agg(mean_absolute_correlation=("absolute_correlation", "mean"))
        .sort_values("mean_absolute_correlation", ascending=False)
    )


def _dashboard_forecasts(ax, comparison: pd.DataFrame) -> None:
    summary = _forecast_summary(comparison).head(6)
    if summary.empty:
        ax.axis("off")
        ax.set_title("Forecasts unavailable")
        return
    ax.barh(summary["model"], summary["mean_rmse"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_title("Forecast RMSE")
    ax.grid(True, axis="x", alpha=0.2)


def _dashboard_features(ax, global_importance: pd.DataFrame) -> None:
    value = "mean_permutation_importance_mean"
    if global_importance.empty or value not in global_importance:
        ax.axis("off")
        ax.set_title("Importance unavailable")
        return
    frame = global_importance.head(6).sort_values(value)
    labels = frame["feature_base"].astype(str)
    ax.barh(labels, frame[value], color="#54A24B")
    ax.set_title("Feature Importance")
    ax.grid(True, axis="x", alpha=0.2)


def _dashboard_abm(ax, comparison: pd.DataFrame) -> None:
    if comparison.empty or "final_economic_output_proxy" not in comparison:
        ax.axis("off")
        ax.set_title("ABM unavailable")
        return
    ax.bar(comparison["scenario"], comparison["final_economic_output_proxy"], color="#F58518")
    ax.set_title("ABM Output Proxy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.2)


def _dashboard_bayes(ax, scenario_probabilities: pd.DataFrame) -> None:
    if scenario_probabilities.empty or "posterior_probability" not in scenario_probabilities:
        ax.axis("off")
        ax.set_title("Bayesian unavailable")
        return
    ax.bar(scenario_probabilities["scenario"], scenario_probabilities["posterior_probability"], color="#B279A2")
    ax.set_title("Scenario Stress Weights")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.2)


def _dashboard_monte_carlo(ax, summary: pd.DataFrame) -> None:
    frame = summary.loc[summary["metric"] == "final_economic_output_proxy"] if not summary.empty else pd.DataFrame()
    if frame.empty:
        ax.axis("off")
        ax.set_title("Monte Carlo unavailable")
        return
    x = np.arange(len(frame))
    yerr = np.vstack([frame["mean"] - frame["q05"], frame["q95"] - frame["mean"]])
    ax.errorbar(x, frame["mean"], yerr=yerr, fmt="o", color="#E45756", capsize=3)
    ax.set_xticks(x, frame["scenario"], rotation=35, ha="right")
    ax.set_title("MC Output Proxy 90% Band")
    ax.grid(True, axis="y", alpha=0.2)


def _dashboard_data_coverage(ax, macro: pd.DataFrame) -> None:
    if macro.empty:
        ax.axis("off")
        ax.set_title("Data unavailable")
        return
    coverage = macro.groupby("country")["value"].apply(lambda s: s.notna().sum()).sort_index()
    ax.bar(coverage.index, coverage.values, color="#72B7B2")
    ax.set_title("Observed Macro Values")
    ax.grid(True, axis="y", alpha=0.2)


if __name__ == "__main__":
    main()
