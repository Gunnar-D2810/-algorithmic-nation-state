"""Run Monte Carlo scenario uncertainty propagation for the ECAIF ABM."""

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

from src.abm.config import load_project_config
from src.monte_carlo.distributions import (
    default_uncertainty_distributions,
    distributions_to_frame,
)
from src.monte_carlo.sensitivity_analysis import rank_sensitivity
from src.monte_carlo.simulation_engine import MonteCarloResult, run_monte_carlo

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo ABM uncertainty propagation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/indicators.yaml",
        help="Path to project YAML config.",
    )
    parser.add_argument(
        "--macro-panel",
        type=Path,
        default=PROJECT_ROOT / "data/processed/macro_panel.csv",
        help="Path to cleaned macro panel CSV.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/tables/probabilistic",
        help="Directory for probabilistic tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/figures/probabilistic",
        help="Directory for probabilistic figures.",
    )
    parser.add_argument(
        "--methodology",
        type=Path,
        default=PROJECT_ROOT / "reports/probabilistic_methodology.md",
        help="Path for probabilistic methodology markdown.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Monte Carlo iterations. Defaults to min(config value, 100) for first-version runtime.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--time-steps", type=int, default=30, help="ABM timesteps per run.")
    parser.add_argument(
        "--skip-reproducibility-check",
        action="store_true",
        help="Skip the compact fixed-seed reproducibility check.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    """Run Monte Carlo pipeline and write artifacts."""

    configure_logging()
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    distributions = default_uncertainty_distributions()
    iterations = args.iterations or _default_iterations(args.config)
    LOGGER.info(
        "Running Monte Carlo uncertainty propagation: iterations=%s, time_steps=%s.",
        iterations,
        args.time_steps,
    )
    result = run_monte_carlo(
        config_path=args.config,
        macro_panel_path=args.macro_panel,
        iterations=iterations,
        seed=args.seed,
        time_steps=args.time_steps,
        distributions=distributions,
    )
    sensitivity = rank_sensitivity(result.runs)
    reproducibility = (
        pd.DataFrame(
            [
                {
                    "check": "fixed_seed_monte_carlo_reproducibility",
                    "passed": True,
                    "detail": "skipped_by_user",
                }
            ]
        )
        if args.skip_reproducibility_check
        else run_reproducibility_check(args, distributions)
    )

    write_tables(result, sensitivity, reproducibility, distributions, args.tables_dir)
    plot_uncertainty_fan(result, args.figures_dir)
    plot_distribution_histograms(result, args.figures_dir)
    plot_sensitivity_tornado(sensitivity, args.figures_dir)
    write_methodology(
        methodology_path=args.methodology,
        distributions=distributions,
        iterations=iterations,
        time_steps=args.time_steps,
        diagnostics=result.diagnostics,
        reproducibility=reproducibility,
    )

    diagnostics = pd.concat([result.diagnostics, reproducibility], ignore_index=True)
    failed_diagnostics = diagnostics.loc[~diagnostics["passed"].astype(bool)]
    if not failed_diagnostics.empty:
        LOGGER.warning(
            "Monte Carlo diagnostics warnings: %s",
            failed_diagnostics.to_dict("records"),
        )
    LOGGER.info("Monte Carlo outputs written to %s and %s", args.tables_dir, args.figures_dir)


def _default_iterations(config_path: Path) -> int:
    """Return a bounded first-version default from the project config."""

    config = load_project_config(config_path)
    configured = int(config.get("simulation", {}).get("monte_carlo", {}).get("iterations", 100))
    return min(configured, 100)


def run_reproducibility_check(
    args: argparse.Namespace,
    distributions,
) -> pd.DataFrame:
    """Run a compact deterministic check with the same seed and parameters."""

    check_iterations = 3
    first = run_monte_carlo(
        config_path=args.config,
        macro_panel_path=args.macro_panel,
        iterations=check_iterations,
        seed=args.seed,
        time_steps=args.time_steps,
        distributions=distributions,
    )
    second = run_monte_carlo(
        config_path=args.config,
        macro_panel_path=args.macro_panel,
        iterations=check_iterations,
        seed=args.seed,
        time_steps=args.time_steps,
        distributions=distributions,
    )
    runs_equal = first.runs.round(10).equals(second.runs.round(10))
    timeseries_equal = first.timeseries.round(10).equals(second.timeseries.round(10))
    return pd.DataFrame(
        [
            {
                "check": "fixed_seed_monte_carlo_reproducibility",
                "passed": bool(runs_equal and timeseries_equal),
                "detail": (
                    "compact_check_identical"
                    if runs_equal and timeseries_equal
                    else "compact_check_mismatch"
                ),
            }
        ]
    )


def write_tables(
    result: MonteCarloResult,
    sensitivity: pd.DataFrame,
    reproducibility: pd.DataFrame,
    distributions,
    tables_dir: Path,
) -> None:
    """Write Monte Carlo tables and root compatibility copies."""

    root_tables_dir = tables_dir.parent
    result.summary.to_csv(tables_dir / "monte_carlo_summary.csv", index=False)
    result.runs.to_csv(tables_dir / "monte_carlo_runs.csv", index=False)
    result.timeseries.to_csv(tables_dir / "monte_carlo_timeseries.csv", index=False)
    result.failures.to_csv(tables_dir / "monte_carlo_failed_runs.csv", index=False)
    result.sampled_parameters.to_csv(
        tables_dir / "monte_carlo_sampled_parameters.csv",
        index=False,
    )
    result.diagnostics.to_csv(tables_dir / "monte_carlo_diagnostics.csv", index=False)
    reproducibility.to_csv(
        tables_dir / "monte_carlo_reproducibility_check.csv",
        index=False,
    )
    sensitivity.to_csv(tables_dir / "sensitivity_rankings.csv", index=False)
    distributions_to_frame(distributions).to_csv(
        tables_dir / "monte_carlo_parameter_distributions.csv",
        index=False,
    )

    result.summary.to_csv(root_tables_dir / "monte_carlo_summary.csv", index=False)
    sensitivity.to_csv(root_tables_dir / "sensitivity_rankings.csv", index=False)


def plot_uncertainty_fan(result: MonteCarloResult, figures_dir: Path) -> None:
    """Plot an uncertainty fan chart for simulated economic output."""

    frame = result.timeseries
    metric = "economic_output_proxy"
    if frame.empty or metric not in frame.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario, scenario_frame in frame.groupby("scenario"):
        quantiles = (
            scenario_frame.groupby("timestep")[metric]
            .quantile([0.05, 0.50, 0.95])
            .unstack()
            .sort_index()
        )
        timesteps = quantiles.index.to_numpy(dtype=float)
        lower = quantiles[0.05].to_numpy(dtype=float)
        median = quantiles[0.50].to_numpy(dtype=float)
        upper = quantiles[0.95].to_numpy(dtype=float)
        ax.plot(timesteps, median, label=scenario)
        ax.fill_between(timesteps, lower, upper, alpha=0.12)
    ax.set_title("Monte Carlo Uncertainty Fan: Economic Output Proxy")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "uncertainty_fan_economic_output_proxy.png", dpi=150)
    plt.close(fig)


def plot_distribution_histograms(result: MonteCarloResult, figures_dir: Path) -> None:
    """Plot Monte Carlo final metric histograms."""

    histogram_dir = figures_dir / "monte_carlo_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "final_compute_concentration_hhi",
        "final_ai_capability_divergence",
        "final_economic_output_proxy",
        "final_dependency_score",
    ]
    for metric in metrics:
        if result.runs.empty or metric not in result.runs.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for scenario, frame in result.runs.groupby("scenario"):
            ax.hist(frame[metric], bins=18, alpha=0.35, label=scenario)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel(metric)
        ax.set_ylabel("Run count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(histogram_dir / f"{metric}_histogram.png", dpi=150)
        plt.close(fig)


def plot_sensitivity_tornado(
    sensitivity: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Plot a tornado-style sensitivity ranking for capability divergence."""

    if sensitivity.empty:
        return
    outcome = "final_ai_capability_divergence"
    frame = (
        sensitivity.loc[sensitivity["outcome"] == outcome]
        .dropna(subset=["absolute_correlation"])
        .sort_values("absolute_correlation", ascending=False)
        .head(14)
        .copy()
    )
    if frame.empty:
        return
    frame["label"] = frame["scenario"] + " | " + frame["parameter"]
    frame = frame.sort_values("absolute_correlation")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(frame["correlation"] >= 0, "#4C78A8", "#F58518")
    ax.barh(frame["label"], frame["correlation"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Sensitivity Tornado: AI Capability Divergence")
    ax.set_xlabel("Spearman correlation")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_tornado_ai_capability_divergence.png", dpi=150)
    plt.close(fig)


def write_methodology(
    *,
    methodology_path: Path,
    distributions,
    iterations: int,
    time_steps: int,
    diagnostics: pd.DataFrame,
    reproducibility: pd.DataFrame,
) -> None:
    """Write probabilistic methodology and limitations documentation."""

    methodology_path.parent.mkdir(parents=True, exist_ok=True)
    distribution_rows = "\n".join(
        [
            (
                f"- `{distribution.name}`: {distribution.distribution}, mean={distribution.mean}, "
                f"range=[{distribution.lower}, {distribution.upper}]. {distribution.notes}"
            )
            for distribution in distributions
        ]
    )
    diagnostic_rows = "\n".join(
        [
            f"- `{row.check}`: passed={row.passed}; {row.detail}"
            for row in pd.concat([diagnostics, reproducibility], ignore_index=True).itertuples(index=False)
        ]
    )
    methodology = f"""# Probabilistic Forecasting And Uncertainty Methodology

This document describes the first exploratory probabilistic layer for the
Algorithmic Nation-State project. It combines closed-form Bayesian updates with
Monte Carlo scenario perturbations around the ECAIF ABM. It is not a calibrated
geopolitical prediction system.

## Bayesian Updating

The Bayesian layer uses conjugate Beta-Binomial updates for binary event
evidence and Dirichlet updates for scenario weights. Priors are weak and
symmetric when no calibration source is supplied. The implementation avoids MCMC
for this first version, so convergence diagnostics are not applicable; posterior
consistency is checked by validating the closed-form arithmetic.

The event examples currently include:

- compute concentration increases
- trade fragmentation intensifies
- AI capability divergence widens
- cloud fragmentation increases
- semiconductor bottlenecks persist
- compute shortages persist

Evidence comes from existing ABM tables and forecast comparison tables. These
are evidence proxies from simulations and model diagnostics, not empirical
event-frequency observations.

## Scenario Probability Adjustments

Scenario probability evolution uses a symmetric Dirichlet prior and cumulative
ABM shock counts as stress evidence. This should be interpreted as an
assumption-transparent scenario-weighting exercise, not a measured probability
that a scenario will occur.

## Monte Carlo Uncertainty Propagation

The Monte Carlo layer samples scenario perturbations and reruns the ABM under
the configured scenarios:

- baseline_globalization
- fragmented_mercantilism
- compute_cold_war
- cooperative_equilibrium

Current runtime used `{iterations}` iterations and `{time_steps}` timesteps per
scenario run.

## Uncertainty Sources

{distribution_rows}

These ranges are subjective first-version stress-test assumptions. They are not
calibrated to observed AI infrastructure, finance, energy, or semiconductor
supply-chain data.

## Sensitivity Analysis

Sensitivity rankings use Spearman rank correlations between sampled parameters
and final ABM outcomes. This is a transparent screening diagnostic, not a causal
decomposition.

## Validation

{diagnostic_rows}

## Interpretation Guidance

- Treat posteriors as conditional on weak priors and simulation-derived evidence
  proxies.
- Treat Monte Carlo outputs as uncertainty propagation through stated
  assumptions.
- Do not describe scenario weights as real-world geopolitical probabilities.
- Do not compare simulated economic output proxies to GDP.
- Do not make causal claims from sensitivity correlations.

## Calibration Limitations

- No external AI infrastructure dataset is used for calibration.
- Shock probabilities and intensities remain scenario assumptions.
- Taiwan uses median-based ABM initialization because the current World Bank
  panel has no observed Taiwan rows.
- Annual macroeconomic data are small and noisy for forecast validation.
- Monte Carlo distributions are inspectable stress-test ranges, not empirical
  distributions.

## Future Extensions

- Add optional calibration files for priors and parameter distributions.
- Calibrate compute and energy constraints using external infrastructure data.
- Add posterior predictive checks once calibrated likelihoods exist.
- Compare Monte Carlo sensitivity rankings across alternative ABM mechanisms.
"""
    methodology_path.write_text(methodology, encoding="utf-8")


if __name__ == "__main__":
    main()
