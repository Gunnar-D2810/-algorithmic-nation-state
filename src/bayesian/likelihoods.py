"""Likelihood evidence construction for Bayesian forecast updates.

The functions here translate existing model outputs into simple success/trial
counts. These counts are evidence proxies from forecasts and simulations, not
validated measurements of geopolitical probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvidenceRecord:
    """Binary evidence summary for one posterior update row."""

    event_name: str
    scope_type: str
    scope_value: str
    successes: float
    trials: float
    evidence_source: str
    notes: str


def validate_evidence(records: list[EvidenceRecord]) -> None:
    """Validate success/trial evidence before posterior updating."""

    for record in records:
        if record.trials < 0:
            raise ValueError(f"Negative trials for {record}.")
        if record.successes < 0:
            raise ValueError(f"Negative successes for {record}.")
        if record.successes > record.trials:
            raise ValueError(f"Successes exceed trials for {record}.")


def scenario_event_evidence(
    metrics: pd.DataFrame,
    shocks: pd.DataFrame,
    states: pd.DataFrame | None = None,
) -> list[EvidenceRecord]:
    """Build scenario-level evidence from ABM metric and shock outputs."""

    records: list[EvidenceRecord] = []
    if metrics.empty:
        return records

    for scenario, frame in metrics.sort_values("timestep").groupby("scenario"):
        timesteps = int(frame["timestep"].max())
        trials = max(timesteps, 1)
        records.extend(
            [
                _metric_increase_record(
                    frame,
                    scenario=scenario,
                    metric="compute_concentration_hhi",
                    event_name="compute_concentration_increases",
                    notes="Success counts are timesteps where ABM compute HHI rises from the prior timestep.",
                ),
                _metric_increase_record(
                    frame,
                    scenario=scenario,
                    metric="ai_capability_divergence",
                    event_name="ai_capability_divergence_widens",
                    notes=(
                        "Success counts are timesteps where simulated firm capability "
                        "dispersion rises from the prior timestep."
                    ),
                ),
            ]
        )

        scenario_shocks = _scenario_shocks(shocks, scenario)
        records.append(
            _shock_record(
                scenario_shocks,
                scenario=scenario,
                trials=trials,
                event_name="cloud_fragmentation_increases",
                shock_types={"cloud_fragmentation"},
                notes="Success counts are timesteps with an ABM cloud-fragmentation shock.",
            )
        )
        records.append(
            _shock_record(
                scenario_shocks,
                scenario=scenario,
                trials=trials,
                event_name="semiconductor_bottlenecks_persist",
                shock_types={"semiconductor_export_restrictions", "compute_shortages"},
                notes=(
                    "Success counts are timesteps with semiconductor export "
                    "restrictions or compute shortages in the ABM shock log."
                ),
            )
        )
        records.append(
            _shock_record(
                scenario_shocks,
                scenario=scenario,
                trials=trials,
                event_name="compute_shortage_persists",
                shock_types={"compute_shortages"},
                notes="Success counts are timesteps with an ABM compute-shortage shock.",
            )
        )
        records.append(
            _trade_fragmentation_record(
                states,
                scenario=scenario,
                fallback_shocks=scenario_shocks,
                trials=trials,
            )
        )
    return records


def model_performance_evidence(comparison: pd.DataFrame) -> list[EvidenceRecord]:
    """Build model-level reliability evidence from forecast comparison tables."""

    if comparison.empty or not {"country", "model", "rmse"}.issubset(comparison.columns):
        return []

    frame = comparison.copy()
    frame["rmse"] = pd.to_numeric(frame["rmse"], errors="coerce")
    frame = frame.loc[frame["rmse"].notna()]
    if frame.empty:
        return []

    country_medians = frame.groupby("country")["rmse"].transform("median")
    frame["success"] = frame["rmse"] <= country_medians
    notes = "Success means model RMSE is at or below the country median among available baselines."
    if "underperforms_naive" in frame.columns and frame["underperforms_naive"].notna().any():
        has_naive_diagnostic = frame["underperforms_naive"].notna()
        underperforms = frame.loc[has_naive_diagnostic, "underperforms_naive"].map(_as_bool)
        frame.loc[has_naive_diagnostic, "success"] = ~underperforms
        notes = (
            "Success uses the naive-baseline diagnostic where available; rows "
            "without that diagnostic fall back to country-median RMSE."
        )

    records: list[EvidenceRecord] = []
    for model, group in frame.groupby("model"):
        records.append(
            EvidenceRecord(
                event_name="model_forecast_reliability",
                scope_type="model",
                scope_value=str(model),
                successes=float(group["success"].sum()),
                trials=float(len(group)),
                evidence_source="reports/tables/all_model_comparison.csv",
                notes=notes,
            )
        )
    return records


def country_policy_stress_evidence(states: pd.DataFrame) -> list[EvidenceRecord]:
    """Build country-level evidence from ABM state policy-pressure trajectories."""

    required = {
        "country",
        "scenario",
        "timestep",
        "export_restriction",
        "data_localization",
        "capital_control",
    }
    if states.empty or not required.issubset(states.columns):
        return []

    frame = states.copy()
    policy_columns = ["export_restriction", "data_localization", "capital_control"]
    frame["policy_stress"] = frame[policy_columns].sum(axis=1)
    records: list[EvidenceRecord] = []

    for country, country_frame in frame.sort_values("timestep").groupby("country"):
        successes = 0
        trials = 0
        for _, scenario_frame in country_frame.groupby("scenario"):
            diffs = scenario_frame.sort_values("timestep")["policy_stress"].diff().dropna()
            successes += int((diffs > 0).sum())
            trials += int(len(diffs))
        records.append(
            EvidenceRecord(
                event_name="country_policy_stress_increases",
                scope_type="country",
                scope_value=str(country),
                successes=float(successes),
                trials=float(max(trials, 1)),
                evidence_source="reports/tables/abm/abm_state_states.csv",
                notes=(
                    "Success counts are country-scenario timesteps where ABM "
                    "export, data-localization, or capital-control pressure rises."
                ),
            )
        )
    return records


def _metric_increase_record(
    frame: pd.DataFrame,
    *,
    scenario: str,
    metric: str,
    event_name: str,
    notes: str,
) -> EvidenceRecord:
    values = pd.to_numeric(frame[metric], errors="coerce")
    diffs = values.diff().dropna()
    return EvidenceRecord(
        event_name=event_name,
        scope_type="scenario",
        scope_value=str(scenario),
        successes=float((diffs > 0).sum()),
        trials=float(max(len(diffs), 1)),
        evidence_source="reports/tables/abm/abm_timeseries.csv",
        notes=notes,
    )


def _scenario_shocks(shocks: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if shocks.empty or "scenario" not in shocks.columns:
        return pd.DataFrame()
    return shocks.loc[shocks["scenario"] == scenario].copy()


def _shock_record(
    shocks: pd.DataFrame,
    *,
    scenario: str,
    trials: int,
    event_name: str,
    shock_types: set[str],
    notes: str,
) -> EvidenceRecord:
    if shocks.empty or "shock_type" not in shocks.columns:
        successes = 0
    else:
        successes = shocks.loc[shocks["shock_type"].isin(shock_types), "timestep"].nunique()
    return EvidenceRecord(
        event_name=event_name,
        scope_type="scenario",
        scope_value=str(scenario),
        successes=float(successes),
        trials=float(max(trials, 1)),
        evidence_source="reports/tables/abm/abm_shock_events.csv",
        notes=notes,
    )


def _trade_fragmentation_record(
    states: pd.DataFrame | None,
    *,
    scenario: str,
    fallback_shocks: pd.DataFrame,
    trials: int,
) -> EvidenceRecord:
    if states is not None and not states.empty and "trade_openness" in states.columns:
        scenario_states = states.loc[states["scenario"] == scenario].copy()
        mean_trade = (
            scenario_states.groupby("timestep")["trade_openness"]
            .mean()
            .sort_index()
        )
        diffs = mean_trade.diff().dropna()
        successes = float((diffs < 0).sum())
        trials = int(max(len(diffs), 1))
        source = "reports/tables/abm/abm_state_states.csv"
        notes = "Success counts are timesteps where mean simulated state trade openness declines."
    else:
        shock_types = {"tariffs", "cloud_fragmentation", "data_localization"}
        successes = (
            fallback_shocks.loc[fallback_shocks["shock_type"].isin(shock_types), "timestep"].nunique()
            if not fallback_shocks.empty
            else 0
        )
        source = "reports/tables/abm/abm_shock_events.csv"
        notes = "Fallback success counts use tariff, cloud-fragmentation, or data-localization shock timesteps."

    return EvidenceRecord(
        event_name="trade_fragmentation_intensifies",
        scope_type="scenario",
        scope_value=str(scenario),
        successes=float(np.clip(successes, 0, trials)),
        trials=float(max(trials, 1)),
        evidence_source=source,
        notes=notes,
    )


def _as_bool(value) -> bool:
    """Parse boolean-like values from CSV diagnostics."""

    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
