"""Simulation orchestration for the ECAIF ABM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.abm.config import ABMConfig, ScenarioConfig
from src.abm.environment import SimulationEnvironment
from src.abm.metrics import compute_metrics, scenario_summary


@dataclass
class SimulationResult:
    """Tabular outputs from one or more ABM scenario runs."""

    metrics: pd.DataFrame
    firms: pd.DataFrame
    providers: pd.DataFrame
    states: pd.DataFrame
    shocks: pd.DataFrame
    networks: pd.DataFrame
    validation: pd.DataFrame
    scenario_comparison: pd.DataFrame
    initialization: pd.DataFrame


def run_scenario(config: ABMConfig, scenario: ScenarioConfig, seed: int) -> SimulationResult:
    """Run one scenario and return all output tables."""

    rng = np.random.default_rng(seed)
    environment = SimulationEnvironment(config=config, scenario=scenario, rng=rng)
    environment.initialize()

    metrics_rows: list[dict] = [
        compute_metrics(
            scenario=scenario.name,
            timestep=0,
            states=environment.states,
            firms=environment.firms,
            providers=environment.providers,
            shock_count=0,
        )
    ]
    state_rows, firm_rows, provider_rows = environment.snapshot_records()
    all_state_rows = state_rows
    all_firm_rows = firm_rows
    all_provider_rows = provider_rows

    for _ in range(config.time_steps):
        metrics_rows.append(environment.step())
        state_rows, firm_rows, provider_rows = environment.snapshot_records()
        all_state_rows.extend(state_rows)
        all_firm_rows.extend(firm_rows)
        all_provider_rows.extend(provider_rows)

    scenario_summary_rows = scenario_summary(metrics_rows)
    return SimulationResult(
        metrics=pd.DataFrame(metrics_rows),
        firms=pd.DataFrame(all_firm_rows),
        providers=pd.DataFrame(all_provider_rows),
        states=pd.DataFrame(all_state_rows),
        shocks=pd.DataFrame(environment.shock_records()),
        networks=pd.DataFrame(environment.network_edge_records),
        validation=pd.DataFrame(environment.validation_records),
        scenario_comparison=pd.DataFrame(scenario_summary_rows),
        initialization=initialization_table(config),
    )


def run_all_scenarios(config: ABMConfig) -> SimulationResult:
    """Run all configured scenarios with deterministic scenario seeds."""

    results: list[SimulationResult] = []
    for index, scenario in enumerate(config.scenarios):
        results.append(run_scenario(config, scenario, seed=config.seed + 1009 * index))
    return combine_results(results)


def combine_results(results: list[SimulationResult]) -> SimulationResult:
    """Combine scenario result tables."""

    return SimulationResult(
        metrics=pd.concat([result.metrics for result in results], ignore_index=True),
        firms=pd.concat([result.firms for result in results], ignore_index=True),
        providers=pd.concat([result.providers for result in results], ignore_index=True),
        states=pd.concat([result.states for result in results], ignore_index=True),
        shocks=pd.concat(
            [result.shocks for result in results if not result.shocks.empty],
            ignore_index=True,
        )
        if any(not result.shocks.empty for result in results)
        else pd.DataFrame(),
        networks=pd.concat(
            [result.networks for result in results if not result.networks.empty],
            ignore_index=True,
        )
        if any(not result.networks.empty for result in results)
        else pd.DataFrame(),
        validation=pd.concat([result.validation for result in results], ignore_index=True),
        scenario_comparison=pd.concat(
            [result.scenario_comparison for result in results],
            ignore_index=True,
        ),
        initialization=results[0].initialization if results else pd.DataFrame(),
    )


def initialization_table(config: ABMConfig) -> pd.DataFrame:
    """Return transparent country initialization assumptions."""

    return pd.DataFrame(
        [
            {
                "country": country.code,
                "name": country.name,
                "compute_base": country.compute_base,
                "data_base": country.data_base,
                "capital_base": country.capital_base,
                "energy_base": country.energy_base,
                "talent_base": country.talent_base,
                "trade_openness": country.trade_openness,
                "industrial_policy": country.industrial_policy,
                "empirical_support": country.empirical_support,
            }
            for country in config.countries
        ]
    )


def verify_reproducibility(config: ABMConfig) -> dict[str, str | bool]:
    """Run a compact reproducibility check for the first scenario."""

    scenario = config.scenarios[0]
    first = run_scenario(config, scenario, seed=config.seed)
    second = run_scenario(config, scenario, seed=config.seed)
    same_metrics = first.metrics.round(10).equals(second.metrics.round(10))
    same_firms = first.firms.round(10).equals(second.firms.round(10))
    return {
        "check": "fixed_seed_reproducibility",
        "scenario": scenario.name,
        "passed": bool(same_metrics and same_firms),
        "detail": "metrics_and_firm_tables_identical" if same_metrics and same_firms else "mismatch_detected",
    }


def write_result_tables(result: SimulationResult, output_dir) -> None:
    """Write ABM result tables to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output_dir / "abm_timeseries.csv", index=False)
    result.firms.to_csv(output_dir / "abm_firm_states.csv", index=False)
    result.providers.to_csv(output_dir / "abm_provider_states.csv", index=False)
    result.states.to_csv(output_dir / "abm_state_states.csv", index=False)
    result.shocks.to_csv(output_dir / "abm_shock_events.csv", index=False)
    result.networks.to_csv(output_dir / "abm_network_edges.csv", index=False)
    result.validation.to_csv(output_dir / "abm_validation_report.csv", index=False)
    result.scenario_comparison.to_csv(output_dir / "abm_scenario_comparison.csv", index=False)
    result.initialization.to_csv(output_dir / "abm_initialization.csv", index=False)
