"""Scenario perturbation helpers for Monte Carlo ABM runs."""

from __future__ import annotations

from dataclasses import replace

from src.abm.config import ABMConfig, ScenarioConfig
from src.abm.resources import bounded


def perturb_scenario(
    scenario: ScenarioConfig,
    parameters: dict[str, float],
) -> ScenarioConfig:
    """Return a scenario with transparent Monte Carlo perturbations applied."""

    compute_supply = parameters["compute_supply_multiplier"]
    capital_access = parameters["capital_access_multiplier"]
    energy_access = parameters["energy_constraint_multiplier"]
    export_intensity = parameters["export_control_intensity"]
    rd_acceleration = parameters["r_and_d_acceleration_multiplier"]
    productivity = parameters["ai_productivity_multiplier"]
    data_fragmentation = parameters["data_fragmentation_intensity"]

    trade_multiplier = scenario.trade_openness_multiplier / (
        1.0 + max(export_intensity - 1.0, 0.0) * 0.18
    )
    data_sharing = scenario.data_sharing / (
        1.0 + max(data_fragmentation - 1.0, 0.0) * 0.45
    )
    return replace(
        scenario,
        trade_openness_multiplier=max(0.05, trade_multiplier),
        shock_probability=bounded(
            scenario.shock_probability * export_intensity * data_fragmentation,
            lower=0.0,
            upper=0.95,
        ),
        shock_intensity=bounded(
            scenario.shock_intensity * export_intensity,
            lower=0.01,
            upper=0.98,
        ),
        compute_growth_rate=max(
            0.0,
            scenario.compute_growth_rate * compute_supply * energy_access,
        ),
        capital_mobility=bounded(
            scenario.capital_mobility * capital_access,
            lower=0.02,
            upper=1.25,
        ),
        data_sharing=bounded(data_sharing, lower=0.02, upper=1.25),
        r_and_d_reinvestment=bounded(
            scenario.r_and_d_reinvestment * rd_acceleration * productivity,
            lower=0.01,
            upper=0.95,
        ),
        compute_demand_factor=max(0.20, scenario.compute_demand_factor * productivity),
    )


def build_perturbed_config(
    base_config: ABMConfig,
    scenario: ScenarioConfig,
    parameters: dict[str, float],
    *,
    seed: int,
) -> ABMConfig:
    """Create a single-scenario ABM config for one Monte Carlo draw."""

    return replace(
        base_config,
        seed=seed,
        scenarios=(perturb_scenario(scenario, parameters),),
    )
