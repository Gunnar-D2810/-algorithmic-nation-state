"""Configuration and initialization helpers for the ECAIF ABM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.abm.resources import normalized


@dataclass(frozen=True)
class CountryInitialization:
    """Country-level initial conditions for the ABM."""

    code: str
    name: str
    compute_base: float
    data_base: float
    capital_base: float
    energy_base: float
    talent_base: float
    trade_openness: float
    industrial_policy: float
    empirical_support: str


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario assumptions for mercantilist AI competition."""

    name: str
    trade_openness_multiplier: float
    cooperation_factor: float
    shock_probability: float
    shock_intensity: float
    compute_growth_rate: float
    capital_mobility: float
    data_sharing: float
    r_and_d_reinvestment: float
    compute_demand_factor: float
    targeted_fragmentation: bool
    description: str


@dataclass(frozen=True)
class ABMConfig:
    """Top-level ABM runtime configuration."""

    seed: int
    time_steps: int
    firms_per_country: int
    providers_per_country: int
    scenarios: tuple[ScenarioConfig, ...]
    countries: tuple[CountryInitialization, ...]


def load_project_config(config_path: Path) -> dict[str, Any]:
    """Load the existing project YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def default_scenarios() -> tuple[ScenarioConfig, ...]:
    """Return transparent conceptual scenarios for the first ABM version."""

    return (
        ScenarioConfig(
            name="baseline_globalization",
            trade_openness_multiplier=1.0,
            cooperation_factor=0.65,
            shock_probability=0.08,
            shock_intensity=0.20,
            compute_growth_rate=0.045,
            capital_mobility=0.85,
            data_sharing=0.70,
            r_and_d_reinvestment=0.18,
            compute_demand_factor=1.15,
            targeted_fragmentation=False,
            description="Open trade and moderate cooperation with occasional shocks.",
        ),
        ScenarioConfig(
            name="fragmented_mercantilism",
            trade_openness_multiplier=0.62,
            cooperation_factor=0.35,
            shock_probability=0.18,
            shock_intensity=0.42,
            compute_growth_rate=0.035,
            capital_mobility=0.55,
            data_sharing=0.38,
            r_and_d_reinvestment=0.20,
            compute_demand_factor=1.20,
            targeted_fragmentation=True,
            description="Higher policy barriers, fragmented data flows, and reduced capital mobility.",
        ),
        ScenarioConfig(
            name="compute_cold_war",
            trade_openness_multiplier=0.45,
            cooperation_factor=0.20,
            shock_probability=0.24,
            shock_intensity=0.58,
            compute_growth_rate=0.030,
            capital_mobility=0.42,
            data_sharing=0.25,
            r_and_d_reinvestment=0.24,
            compute_demand_factor=1.35,
            targeted_fragmentation=True,
            description="Persistent compute rivalry with export controls and semiconductor bottlenecks.",
        ),
        ScenarioConfig(
            name="cooperative_equilibrium",
            trade_openness_multiplier=1.12,
            cooperation_factor=0.82,
            shock_probability=0.05,
            shock_intensity=0.14,
            compute_growth_rate=0.055,
            capital_mobility=0.92,
            data_sharing=0.82,
            r_and_d_reinvestment=0.16,
            compute_demand_factor=1.08,
            targeted_fragmentation=False,
            description="High coordination, shared standards, and lower fragmentation pressure.",
        ),
    )


def build_abm_config(
    *,
    config_path: Path,
    macro_panel_path: Path,
    seed: int = 42,
    time_steps: int = 30,
) -> ABMConfig:
    """Create ABM configuration from project config and macro panel hints."""

    project_config = load_project_config(config_path)
    countries = build_country_initializations(
        project_config.get("countries", []),
        macro_panel_path,
    )
    return ABMConfig(
        seed=seed,
        time_steps=time_steps,
        firms_per_country=2,
        providers_per_country=1,
        scenarios=default_scenarios(),
        countries=tuple(countries),
    )


def build_country_initializations(
    countries: list[dict[str, str]],
    macro_panel_path: Path,
) -> list[CountryInitialization]:
    """Build transparent initial resource assumptions for countries.

    World Bank indicators provide rough scale hints where available. Missing
    country rows, such as Taiwan in the current panel, receive median-based
    conceptual defaults and are explicitly marked as assumption-driven.
    """

    panel = pd.read_csv(macro_panel_path)
    latest = _latest_indicator_table(panel)
    metric_bounds = {
        column: (
            latest[column].dropna().min(),
            latest[column].dropna().max(),
        )
        for column in latest.columns
        if column not in {"country"}
    }
    medians = latest.drop(columns=["country"], errors="ignore").median(numeric_only=True)

    initializations: list[CountryInitialization] = []
    for country in countries:
        code = country["code"]
        name = country.get("name", code)
        row = latest.loc[latest["country"] == code]
        empirical_support = "world_bank_latest_indicators"
        if row.empty:
            values = medians.to_dict()
            empirical_support = "assumption_median_defaults_due_missing_panel"
        else:
            values = row.iloc[0].to_dict()

        def norm(column: str) -> float:
            low, high = metric_bounds.get(column, (0.0, 1.0))
            return normalized(float(values.get(column, medians.get(column, 0.0))), low, high)

        gdp_norm = norm("GDP_PER_CAPITA")
        tech_norm = norm("HIGH_TECH_EXPORTS")
        fdi_norm = norm("FDI_NET_INFLOWS")
        military_norm = norm("MILITARY_EXPENDITURE")
        exports = float(values.get("EXPORTS_PERCENT_GDP", medians.get("EXPORTS_PERCENT_GDP", 20.0)))
        imports = float(values.get("IMPORTS_PERCENT_GDP", medians.get("IMPORTS_PERCENT_GDP", 20.0)))
        trade_norm = normalized((exports + imports) / 2.0, 10.0, 45.0)

        initializations.append(
            CountryInitialization(
                code=code,
                name=name,
                compute_base=45.0 + 45.0 * tech_norm + 20.0 * gdp_norm,
                data_base=45.0 + 35.0 * trade_norm + 20.0 * tech_norm,
                capital_base=45.0 + 60.0 * gdp_norm + 20.0 * fdi_norm,
                energy_base=55.0 + 18.0 * gdp_norm + 15.0 * military_norm,
                talent_base=45.0 + 45.0 * gdp_norm + 15.0 * tech_norm,
                trade_openness=normalized((exports + imports) / 2.0, 10.0, 45.0, default=0.5),
                industrial_policy=0.04 + 0.05 * tech_norm + 0.03 * military_norm,
                empirical_support=empirical_support,
            )
        )
    return initializations


def _latest_indicator_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Create latest country-level indicator table from the macro panel."""

    indicators = [
        "GDP_PER_CAPITA",
        "EXPORTS_PERCENT_GDP",
        "IMPORTS_PERCENT_GDP",
        "FDI_NET_INFLOWS",
        "HIGH_TECH_EXPORTS",
        "MILITARY_EXPENDITURE",
    ]
    latest_rows = (
        panel.loc[panel["indicator"].isin(indicators) & panel["value"].notna()]
        .sort_values("year")
        .groupby(["country", "indicator"], as_index=False)
        .tail(1)
    )
    return latest_rows.pivot_table(
        index="country",
        columns="indicator",
        values="value",
        aggfunc="first",
    ).reset_index()
