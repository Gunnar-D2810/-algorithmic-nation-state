"""Sensitivity rankings for Monte Carlo uncertainty drivers."""

from __future__ import annotations

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = [
    "compute_supply_multiplier",
    "capital_access_multiplier",
    "energy_constraint_multiplier",
    "export_control_intensity",
    "r_and_d_acceleration_multiplier",
    "ai_productivity_multiplier",
    "data_fragmentation_intensity",
]

OUTCOME_COLUMNS = [
    "final_compute_concentration_hhi",
    "final_compute_inequality_gini",
    "final_ai_capability_divergence",
    "final_economic_output_proxy",
    "final_resilience_score",
    "final_dependency_score",
    "final_infrastructure_asymmetry",
    "total_shock_events",
]


def rank_sensitivity(
    runs: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Rank parameter sensitivity using simple rank correlations."""

    if runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for scenario, frame in runs.groupby("scenario"):
        for outcome in OUTCOME_COLUMNS:
            if outcome not in frame.columns:
                continue
            for parameter in PARAMETER_COLUMNS:
                if parameter not in frame.columns:
                    continue
                clean = frame[[parameter, outcome]].dropna()
                if len(clean) < 3 or clean[parameter].nunique() < 2 or clean[outcome].nunique() < 2:
                    corr = float("nan")
                else:
                    corr = float(clean[parameter].corr(clean[outcome], method=method))
                rows.append(
                    {
                        "scenario": scenario,
                        "outcome": outcome,
                        "parameter": parameter,
                        "correlation_method": method,
                        "correlation": corr,
                        "absolute_correlation": abs(corr) if np.isfinite(corr) else float("nan"),
                        "direction": _direction(corr),
                        "n_runs": int(len(clean)),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["scenario", "outcome", "absolute_correlation"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _direction(correlation: float) -> str:
    if not np.isfinite(correlation):
        return "insufficient_variation"
    if correlation > 0:
        return "positive"
    if correlation < 0:
        return "negative"
    return "zero"
