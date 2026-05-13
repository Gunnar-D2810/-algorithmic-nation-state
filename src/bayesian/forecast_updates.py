"""Orchestration for Bayesian forecast and scenario updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.abm.config import default_scenarios
from src.bayesian.likelihoods import (
    EvidenceRecord,
    country_policy_stress_evidence,
    model_performance_evidence,
    scenario_event_evidence,
    validate_evidence,
)
from src.bayesian.posteriors import (
    posterior_dataframe,
    update_beta_prior,
    update_dirichlet_prior,
    validate_posterior_consistency,
    validate_probability_normalization,
)
from src.bayesian.priors import (
    default_event_priors,
    default_scenario_prior,
    default_scope_prior,
)


@dataclass(frozen=True)
class BayesianUpdateResult:
    """Outputs from the probabilistic forecast update pipeline."""

    posteriors: pd.DataFrame
    scenario_probabilities: pd.DataFrame
    scenario_probability_evolution: pd.DataFrame
    diagnostics: pd.DataFrame
    evidence: pd.DataFrame


def run_bayesian_update_pipeline(
    *,
    abm_tables_dir: Path,
    model_comparison_path: Path,
    interval_mass: float = 0.9,
) -> BayesianUpdateResult:
    """Run closed-form Bayesian updates from existing ABM and forecast outputs."""

    metrics = _read_csv(abm_tables_dir / "abm_timeseries.csv")
    shocks = _read_csv(abm_tables_dir / "abm_shock_events.csv")
    states = _read_csv(abm_tables_dir / "abm_state_states.csv")
    model_comparison = _read_csv(model_comparison_path)

    evidence_records: list[EvidenceRecord] = []
    evidence_records.extend(scenario_event_evidence(metrics, shocks, states))
    evidence_records.extend(country_policy_stress_evidence(states))
    evidence_records.extend(model_performance_evidence(model_comparison))
    validate_evidence(evidence_records)

    priors = default_event_priors(
        sorted({record.event_name for record in evidence_records})
    )
    posteriors = [
        update_beta_prior(
            priors.get(record.event_name, default_scope_prior(record.event_name)),
            record,
            interval_mass=interval_mass,
        )
        for record in evidence_records
    ]
    posterior_frame = posterior_dataframe(posteriors)

    scenario_names = _scenario_names(metrics)
    scenario_prior = default_scenario_prior(scenario_names)
    scenario_counts = _scenario_stress_counts(metrics)
    scenario_probabilities = update_dirichlet_prior(scenario_prior, scenario_counts)
    scenario_evolution = build_scenario_probability_evolution(metrics, scenario_names)

    diagnostics = _diagnostics(
        posterior_frame=posterior_frame,
        scenario_probabilities=scenario_probabilities,
        scenario_evolution=scenario_evolution,
        evidence_records=evidence_records,
    )
    return BayesianUpdateResult(
        posteriors=posterior_frame,
        scenario_probabilities=scenario_probabilities,
        scenario_probability_evolution=scenario_evolution,
        diagnostics=diagnostics,
        evidence=_evidence_dataframe(evidence_records),
    )


def build_scenario_probability_evolution(
    metrics: pd.DataFrame,
    scenario_names: list[str],
) -> pd.DataFrame:
    """Build a timestep-by-timestep Dirichlet update from cumulative shock counts."""

    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "timestep",
                "scenario",
                "prior_alpha",
                "cumulative_stress_evidence",
                "posterior_alpha",
                "posterior_probability",
                "notes",
            ]
        )

    max_timestep = int(metrics["timestep"].max())
    prior_alpha = {name: 1.0 for name in scenario_names}
    rows: list[dict[str, float | int | str]] = []
    for timestep in range(max_timestep + 1):
        posterior_alpha: dict[str, float] = {}
        for scenario in scenario_names:
            frame = metrics.loc[
                (metrics["scenario"] == scenario) & (metrics["timestep"] <= timestep)
            ]
            cumulative = float(frame["shock_count"].sum()) if "shock_count" in frame else 0.0
            posterior_alpha[scenario] = prior_alpha[scenario] + cumulative
        total = sum(posterior_alpha.values())
        for scenario in scenario_names:
            cumulative = posterior_alpha[scenario] - prior_alpha[scenario]
            rows.append(
                {
                    "timestep": timestep,
                    "scenario": scenario,
                    "prior_alpha": prior_alpha[scenario],
                    "cumulative_stress_evidence": cumulative,
                    "posterior_alpha": posterior_alpha[scenario],
                    "posterior_probability": posterior_alpha[scenario] / total,
                    "notes": (
                        "Scenario probabilities are Dirichlet weights from "
                        "cumulative ABM shock counts, not calibrated forecasts."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _scenario_names(metrics: pd.DataFrame) -> list[str]:
    if not metrics.empty and "scenario" in metrics.columns:
        return sorted(metrics["scenario"].dropna().astype(str).unique().tolist())
    return [scenario.name for scenario in default_scenarios()]


def _scenario_stress_counts(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty or "scenario" not in metrics.columns:
        return {}
    counts: dict[str, float] = {}
    for scenario, frame in metrics.groupby("scenario"):
        counts[str(scenario)] = float(frame.get("shock_count", pd.Series(dtype=float)).sum())
    return counts


def _evidence_dataframe(records: list[EvidenceRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def _diagnostics(
    *,
    posterior_frame: pd.DataFrame,
    scenario_probabilities: pd.DataFrame,
    scenario_evolution: pd.DataFrame,
    evidence_records: list[EvidenceRecord],
) -> pd.DataFrame:
    diagnostic_frames = [validate_posterior_consistency(posterior_frame)]

    scenario_norm = validate_probability_normalization(
        scenario_probabilities.assign(group="final"),
        group_columns=["group"],
    )
    scenario_norm["check"] = "scenario_probability_normalization"
    scenario_norm["detail"] = scenario_norm["probability_sum"].map(
        lambda value: f"sum={value:.12f}"
    )
    diagnostic_frames.append(scenario_norm[["check", "passed", "detail"]])

    evolution_norm = validate_probability_normalization(
        scenario_evolution,
        group_columns=["timestep"],
    )
    if not evolution_norm.empty:
        evolution_norm["check"] = "scenario_probability_evolution_normalization"
        evolution_norm["detail"] = evolution_norm.apply(
            lambda row: f"timestep={row['timestep']};sum={row['probability_sum']:.12f}",
            axis=1,
        )
        diagnostic_frames.append(evolution_norm[["check", "passed", "detail"]])

    diagnostic_frames.append(
        pd.DataFrame(
            [
                {
                    "check": "evidence_records_available",
                    "passed": bool(evidence_records),
                    "detail": f"n_records={len(evidence_records)}",
                },
                {
                    "check": "convergence_diagnostics",
                    "passed": True,
                    "detail": (
                        "not_applicable_closed_form_conjugate_updates_no_mcmc_used"
                    ),
                },
            ]
        )
    )
    return pd.concat(diagnostic_frames, ignore_index=True)
