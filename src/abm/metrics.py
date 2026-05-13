"""Metrics for ECAIF ABM simulations."""

from __future__ import annotations

import numpy as np

from src.abm.agents import AIFirmAgent, ComputeProviderAgent, StateAgent
from src.abm.resources import gini, herfindahl_index, safe_divide


def compute_metrics(
    *,
    scenario: str,
    timestep: int,
    states: list[StateAgent],
    firms: list[AIFirmAgent],
    providers: list[ComputeProviderAgent],
    shock_count: int,
) -> dict[str, float | int | str]:
    """Compute system-level metrics for one timestep."""

    firm_compute = [firm.resources.compute for firm in firms]
    capabilities = np.asarray([firm.capability for firm in firms], dtype=float)
    outputs = [firm.last_output for firm in firms]
    provider_capacity = [provider.capacity for provider in providers]
    dependency_scores = [firm.dependency_score() for firm in firms]
    resilience_scores = [firm.resilience for firm in firms]

    mean_capability = float(np.mean(capabilities)) if len(capabilities) else 0.0
    capability_divergence = safe_divide(float(np.std(capabilities)), mean_capability)
    economic_output_proxy = float(sum(outputs))

    return {
        "scenario": scenario,
        "timestep": timestep,
        "compute_concentration_hhi": herfindahl_index(firm_compute),
        "compute_inequality_gini": gini(firm_compute),
        "ai_capability_divergence": capability_divergence,
        "economic_output_proxy": economic_output_proxy,
        "resilience_score": float(np.mean(resilience_scores)) if resilience_scores else 0.0,
        "dependency_score": float(np.mean(dependency_scores)) if dependency_scores else 0.0,
        "infrastructure_asymmetry": gini(provider_capacity),
        "total_compute_capacity": float(sum(provider_capacity)),
        "total_firm_compute": float(sum(firm_compute)),
        "mean_capability": mean_capability,
        "shock_count": shock_count,
    }


def scenario_summary(metrics_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    """Build final scenario comparison rows from timestep metrics."""

    summaries: list[dict[str, float | str]] = []
    scenario_names = sorted({str(row["scenario"]) for row in metrics_rows})
    for scenario in scenario_names:
        rows = [row for row in metrics_rows if row["scenario"] == scenario]
        final = max(rows, key=lambda row: int(row["timestep"]))
        summaries.append(
            {
                "scenario": scenario,
                "final_compute_concentration_hhi": float(final["compute_concentration_hhi"]),
                "final_compute_inequality_gini": float(final["compute_inequality_gini"]),
                "final_ai_capability_divergence": float(final["ai_capability_divergence"]),
                "final_economic_output_proxy": float(final["economic_output_proxy"]),
                "final_resilience_score": float(final["resilience_score"]),
                "final_dependency_score": float(final["dependency_score"]),
                "final_infrastructure_asymmetry": float(final["infrastructure_asymmetry"]),
                "total_shock_events": float(sum(float(row["shock_count"]) for row in rows)),
            }
        )
    return summaries
