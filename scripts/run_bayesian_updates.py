"""Run Bayesian forecast updates from ABM and forecasting outputs."""

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
from scipy.stats import beta as beta_distribution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bayesian.forecast_updates import BayesianUpdateResult, run_bayesian_update_pipeline

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run closed-form Bayesian updates for scenario uncertainty."
    )
    parser.add_argument(
        "--abm-tables-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/tables/abm",
        help="Directory containing ABM output tables.",
    )
    parser.add_argument(
        "--model-comparison",
        type=Path,
        default=PROJECT_ROOT / "reports/tables/all_model_comparison.csv",
        help="Forecast model comparison table.",
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
        "--interval-mass",
        type=float,
        default=0.9,
        help="Credible interval mass for Beta posterior summaries.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    """Run Bayesian update pipeline and write artifacts."""

    configure_logging()
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running Bayesian forecast updates.")
    result = run_bayesian_update_pipeline(
        abm_tables_dir=args.abm_tables_dir,
        model_comparison_path=args.model_comparison,
        interval_mass=args.interval_mass,
    )
    write_tables(result, args.tables_dir)
    plot_posterior_distributions(result, args.figures_dir)
    plot_scenario_probability_evolution(result, args.figures_dir)

    failed_diagnostics = result.diagnostics.loc[~result.diagnostics["passed"].astype(bool)]
    if not failed_diagnostics.empty:
        LOGGER.warning(
            "Bayesian diagnostics warnings: %s",
            failed_diagnostics.to_dict("records"),
        )
    LOGGER.info("Bayesian outputs written to %s and %s", args.tables_dir, args.figures_dir)


def write_tables(result: BayesianUpdateResult, tables_dir: Path) -> None:
    """Write Bayesian update tables and root compatibility copies."""

    root_tables_dir = tables_dir.parent
    result.posteriors.to_csv(tables_dir / "bayesian_posteriors.csv", index=False)
    result.scenario_probabilities.to_csv(
        tables_dir / "scenario_probability_updates.csv",
        index=False,
    )
    result.scenario_probability_evolution.to_csv(
        tables_dir / "scenario_probability_evolution.csv",
        index=False,
    )
    result.diagnostics.to_csv(tables_dir / "bayesian_diagnostics.csv", index=False)
    result.evidence.to_csv(tables_dir / "bayesian_evidence.csv", index=False)

    result.posteriors.to_csv(root_tables_dir / "bayesian_posteriors.csv", index=False)


def plot_posterior_distributions(
    result: BayesianUpdateResult,
    figures_dir: Path,
) -> None:
    """Plot scenario-level Beta posterior distributions."""

    frame = result.posteriors.loc[result.posteriors["scope_type"] == "scenario"].copy()
    if frame.empty:
        return

    events = frame["event_name"].drop_duplicates().tolist()
    ncols = 2
    nrows = int(np.ceil(len(events) / ncols))
    x_values = np.linspace(0.001, 0.999, 250)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.2 * nrows)))
    axes = np.asarray(axes).reshape(-1)

    for ax, event_name in zip(axes, events):
        event_frame = frame.loc[frame["event_name"] == event_name]
        for row in event_frame.itertuples(index=False):
            pdf = beta_distribution.pdf(
                x_values,
                row.posterior_alpha,
                row.posterior_beta,
            )
            ax.plot(x_values, pdf, label=row.scope_value)
        ax.set_title(event_name.replace("_", " ").title())
        ax.set_xlabel("Event probability")
        ax.set_ylabel("Posterior density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[len(events):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "posterior_distributions.png", dpi=150)
    plt.close(fig)


def plot_scenario_probability_evolution(
    result: BayesianUpdateResult,
    figures_dir: Path,
) -> None:
    """Plot scenario probability evolution from cumulative stress evidence."""

    frame = result.scenario_probability_evolution
    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario, scenario_frame in frame.groupby("scenario"):
        ax.plot(
            scenario_frame["timestep"],
            scenario_frame["posterior_probability"],
            label=scenario,
        )
    ax.set_title("Scenario Probability Evolution From ABM Stress Evidence")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Posterior scenario weight")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "scenario_probability_evolution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
