"""Run the first ECAIF agent-based simulation scenarios."""

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
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.abm.config import ABMConfig, build_abm_config
from src.abm.simulation import (
    SimulationResult,
    run_all_scenarios,
    verify_reproducibility,
    write_result_tables,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run ECAIF ABM scenarios.")
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
        default=PROJECT_ROOT / "reports/tables/abm",
        help="Directory for ABM tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/figures/abm",
        help="Directory for ABM figures.",
    )
    parser.add_argument(
        "--methodology",
        type=Path,
        default=PROJECT_ROOT / "reports/abm_methodology.md",
        help="Path for ABM methodology markdown.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--time-steps", type=int, default=30, help="Simulation timesteps.")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    """Run ABM scenarios and write reproducible artifacts."""

    configure_logging()
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    config = build_abm_config(
        config_path=args.config,
        macro_panel_path=args.macro_panel,
        seed=args.seed,
        time_steps=args.time_steps,
    )
    LOGGER.info(
        "Running %s ABM scenarios for %s timesteps.",
        len(config.scenarios),
        config.time_steps,
    )
    result = run_all_scenarios(config)
    reproducibility = verify_reproducibility(config)
    reproducibility_frame = pd.DataFrame([reproducibility])

    write_result_tables(result, args.tables_dir)
    reproducibility_frame.to_csv(
        args.tables_dir / "abm_reproducibility_check.csv",
        index=False,
    )
    plot_metric_timeseries(result, args.figures_dir)
    plot_scenario_comparison(result, args.figures_dir)
    plot_network_graphs(result, args.figures_dir)
    write_methodology(args.methodology, config, reproducibility)

    invalid_warnings = result.validation.loc[result.validation["warning_count"] > 0]
    if not invalid_warnings.empty:
        LOGGER.warning(
            "Validation warnings detected: %s",
            invalid_warnings.head(10).to_dict("records"),
        )
    if not bool(reproducibility["passed"]):
        LOGGER.warning("Reproducibility check failed: %s", reproducibility)
    LOGGER.info("ABM outputs written to %s and %s", args.tables_dir, args.figures_dir)


def plot_metric_timeseries(result: SimulationResult, figures_dir: Path) -> None:
    """Plot key ABM metrics over time."""

    metrics = result.metrics
    metric_names = [
        "compute_concentration_hhi",
        "compute_inequality_gini",
        "ai_capability_divergence",
        "economic_output_proxy",
        "resilience_score",
        "dependency_score",
        "infrastructure_asymmetry",
        "total_compute_capacity",
    ]
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(9, 5))
        for scenario, frame in metrics.groupby("scenario"):
            ax.plot(frame["timestep"], frame[metric], label=scenario)
        ax.set_title(f"ABM {metric.replace('_', ' ').title()}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{metric}.png", dpi=150)
        plt.close(fig)


def plot_scenario_comparison(result: SimulationResult, figures_dir: Path) -> None:
    """Plot final scenario comparison bars."""

    comparison = result.scenario_comparison
    metrics = [
        "final_compute_concentration_hhi",
        "final_ai_capability_divergence",
        "final_economic_output_proxy",
        "final_dependency_score",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.flatten(), metrics):
        ax.bar(comparison["scenario"], comparison[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "scenario_comparison.png", dpi=150)
    plt.close(fig)


def plot_network_graphs(result: SimulationResult, figures_dir: Path) -> None:
    """Plot final conceptual compute-dependency networks by scenario."""

    network_dir = figures_dir / "networks"
    network_dir.mkdir(parents=True, exist_ok=True)
    if result.networks.empty:
        return

    for scenario, frame in result.networks.groupby("scenario"):
        graph = nx.DiGraph()
        for row in frame.itertuples(index=False):
            graph.add_edge(row.source, row.target, relation=row.relation, weight=row.weight)
        node_colors = []
        for node in graph.nodes:
            if str(node).startswith("state"):
                node_colors.append("#4C78A8")
            elif str(node).startswith("provider"):
                node_colors.append("#F58518")
            else:
                node_colors.append("#54A24B")
        fig, ax = plt.subplots(figsize=(11, 8))
        pos = nx.spring_layout(graph, seed=42, k=0.55)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=350, ax=ax)
        nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.25, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=6, ax=ax)
        ax.set_title(f"Final Compute Dependency Network: {scenario}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(network_dir / f"{scenario}_network.png", dpi=150)
        plt.close(fig)


def write_methodology(
    methodology_path: Path,
    config: ABMConfig,
    reproducibility: dict[str, str | bool],
) -> None:
    """Write ABM methodology and limitations documentation."""

    methodology_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_rows = "\n".join(
        [
            f"- `{scenario.name}`: {scenario.description}"
            for scenario in config.scenarios
        ]
    )
    methodology = f"""# ECAIF ABM Methodology

This document describes the first exploratory Evolutionary Cloud AI Firms
(ECAIF) agent-based model. The simulation is a conceptual research tool, not a
geopolitical prediction engine.

## Purpose

The model explores how states, AI firms, and compute providers interact under
modern mercantilist conditions. It is designed to make assumptions inspectable
and to generate scenario dynamics for later Bayesian and Monte Carlo analysis.

## Agents

- `StateAgent`: represents state-level policy pressure, trade openness,
  industrial policy, export restrictions, data localization, and capital
  controls.
- `AIFirmAgent`: represents cloud AI firms that acquire compute, accumulate
  data/capital/talent, reinvest into R&D, and adapt to shocks.
- `ComputeProviderAgent`: represents compute infrastructure providers that
  scale capacity under capital, energy, and shortage constraints.

## Resources

Each agent uses transparent resource stocks:

- compute
- data
- capital
- energy
- talent

The initial resource levels use latest World Bank indicators as rough scale
hints where available. Missing country panels, currently Taiwan, receive
median-based conceptual defaults and are explicitly marked in
`reports/tables/abm/abm_initialization.csv`.

## Scenarios

{scenario_rows}

## Conceptual Mechanisms

- Firms demand compute as a function of capability and capital.
- Providers allocate constrained compute capacity to domestic, allied, and
  foreign firms depending on scenario openness and active shocks.
- Firms update capability through simple production-style assumptions using
  compute, data, capital, and talent.
- States adjust policy restrictions in response to shocks.
- Providers scale infrastructure through capital mobility, state support, and
  compute-shortage penalties.

## Geopolitical Shocks

The model includes seeded stochastic shocks:

- tariffs
- semiconductor export restrictions
- cloud fragmentation
- data localization
- capital controls
- compute shortages

Shock probabilities and intensities are scenario assumptions, not empirical
estimates.

## Metrics

The simulation records:

- compute concentration index
- compute inequality
- AI capability divergence
- economic output proxy
- resilience score
- dependency score
- infrastructure asymmetry

These metrics are simulation diagnostics. They should not be interpreted as
validated empirical measurements.

## Validation

The run records invalid resources, exploding values, and fixed-seed
reproducibility. Current reproducibility check:

- scenario: `{reproducibility["scenario"]}`
- passed: `{reproducibility["passed"]}`
- detail: `{reproducibility["detail"]}`

## Limitations

- The model is not calibrated to observed AI firm data.
- Resource stocks are stylized indexes, not measured physical quantities.
- GDP, trade, high-tech exports, FDI, and military expenditure only inform
  initialization heuristics.
- Network access rules use broad conceptual alliance groups.
- Shocks are generated from scenario assumptions, not forecast probabilities.
- The economic output proxy is not GDP.

## Interpretation Guidance

Use this ABM to compare mechanisms and scenario sensitivities, not to predict
country outcomes. Results should be described as simulated outputs conditional
on stated assumptions.

## Future Extensions

- Bayesian parameter updates for shock probabilities and resource elasticities.
- Monte Carlo sensitivity over scenario assumptions.
- Calibration against external AI infrastructure datasets where available.
- More detailed supply-chain and energy constraints.
"""
    methodology_path.write_text(methodology, encoding="utf-8")


if __name__ == "__main__":
    main()
