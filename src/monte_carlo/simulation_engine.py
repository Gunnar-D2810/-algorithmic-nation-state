"""Monte Carlo engine for propagating scenario uncertainty through the ABM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm.config import build_abm_config
from src.abm.simulation import run_scenario
from src.monte_carlo.distributions import (
    ParameterDistribution,
    default_uncertainty_distributions,
    sample_parameters,
)
from src.monte_carlo.scenario_generator import build_perturbed_config


FINAL_METRIC_COLUMNS = (
    "final_compute_concentration_hhi",
    "final_compute_inequality_gini",
    "final_ai_capability_divergence",
    "final_economic_output_proxy",
    "final_resilience_score",
    "final_dependency_score",
    "final_infrastructure_asymmetry",
    "total_shock_events",
)


@dataclass(frozen=True)
class MonteCarloResult:
    """Outputs from repeated scenario simulations."""

    runs: pd.DataFrame
    timeseries: pd.DataFrame
    failures: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    sampled_parameters: pd.DataFrame


def run_monte_carlo(
    *,
    config_path: Path,
    macro_panel_path: Path,
    iterations: int,
    seed: int,
    time_steps: int,
    scenario_names: set[str] | None = None,
    distributions: tuple[ParameterDistribution, ...] | None = None,
) -> MonteCarloResult:
    """Run repeated ABM simulations with sampled scenario perturbations."""

    if iterations <= 0:
        raise ValueError("Monte Carlo iterations must be positive.")
    rng = np.random.default_rng(seed)
    distributions = distributions or default_uncertainty_distributions()
    base_config = build_abm_config(
        config_path=config_path,
        macro_panel_path=macro_panel_path,
        seed=seed,
        time_steps=time_steps,
    )

    run_rows: list[dict[str, float | int | str | bool]] = []
    time_rows: list[pd.DataFrame] = []
    failure_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int]] = []
    scenarios = [
        scenario
        for scenario in base_config.scenarios
        if scenario_names is None or scenario.name in scenario_names
    ]
    if not scenarios:
        raise ValueError("No scenarios selected for Monte Carlo run.")

    for iteration in range(1, iterations + 1):
        parameters = sample_parameters(distributions, rng)
        parameter_rows.append({"iteration": iteration, **parameters})
        for scenario_index, scenario in enumerate(scenarios):
            scenario_seed = seed + iteration * 10_003 + scenario_index * 1_009
            try:
                perturbed_config = build_perturbed_config(
                    base_config,
                    scenario,
                    parameters,
                    seed=scenario_seed,
                )
                perturbed_scenario = perturbed_config.scenarios[0]
                result = run_scenario(
                    perturbed_config,
                    perturbed_scenario,
                    seed=scenario_seed,
                )
                summary_row = result.scenario_comparison.iloc[0].to_dict()
                validation_warnings = (
                    int(result.validation["warning_count"].sum())
                    if not result.validation.empty
                    else 0
                )
                unstable = _detect_unstable_run(summary_row, validation_warnings)
                run_rows.append(
                    {
                        "iteration": iteration,
                        "scenario": scenario.name,
                        "seed": scenario_seed,
                        **parameters,
                        **summary_row,
                        "validation_warning_count": validation_warnings,
                        "unstable_run": unstable,
                    }
                )
                ts = result.metrics.copy()
                ts["iteration"] = iteration
                ts["seed"] = scenario_seed
                time_rows.append(ts)
            except Exception as exc:  # pragma: no cover - exercised in integration runs
                failure_rows.append(
                    {
                        "iteration": iteration,
                        "scenario": scenario.name,
                        "seed": scenario_seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    runs = pd.DataFrame(run_rows)
    timeseries = pd.concat(time_rows, ignore_index=True) if time_rows else pd.DataFrame()
    failures = pd.DataFrame(failure_rows)
    sampled_parameters = pd.DataFrame(parameter_rows)
    summary = summarize_monte_carlo_runs(runs)
    diagnostics = monte_carlo_diagnostics(
        runs=runs,
        failures=failures,
        requested_iterations=iterations,
        scenario_count=len(scenarios),
    )
    return MonteCarloResult(
        runs=runs,
        timeseries=timeseries,
        failures=failures,
        summary=summary,
        diagnostics=diagnostics,
        sampled_parameters=sampled_parameters,
    )


def summarize_monte_carlo_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize final metric uncertainty by scenario."""

    if runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for (scenario, metric), values in _metric_groups(runs):
        clean = pd.to_numeric(values, errors="coerce").dropna()
        scenario_runs = runs.loc[runs["scenario"] == scenario]
        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "n_runs": int(len(clean)),
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "std": float(clean.std(ddof=0)),
                "q05": float(clean.quantile(0.05)),
                "q95": float(clean.quantile(0.95)),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "unstable_run_rate": float(scenario_runs["unstable_run"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "metric"]).reset_index(drop=True)


def monte_carlo_diagnostics(
    *,
    runs: pd.DataFrame,
    failures: pd.DataFrame,
    requested_iterations: int,
    scenario_count: int,
) -> pd.DataFrame:
    """Return reproducibility and stability diagnostics for the MC run."""

    expected_runs = requested_iterations * scenario_count
    observed_runs = len(runs)
    unstable_count = int(runs["unstable_run"].sum()) if "unstable_run" in runs else 0
    return pd.DataFrame(
        [
            {
                "check": "requested_runs_completed",
                "passed": observed_runs == expected_runs,
                "detail": f"observed={observed_runs};expected={expected_runs};failures={len(failures)}",
            },
            {
                "check": "unstable_simulations_detected",
                "passed": unstable_count == 0,
                "detail": f"unstable_runs={unstable_count}",
            },
            {
                "check": "finite_output_metrics",
                "passed": _all_metric_values_finite(runs),
                "detail": "all_final_metric_values_finite",
            },
            _half_sample_stability_check(runs),
        ]
    )


def _metric_groups(runs: pd.DataFrame):
    for scenario, frame in runs.groupby("scenario"):
        for metric in FINAL_METRIC_COLUMNS:
            if metric in frame.columns:
                yield (str(scenario), metric), frame[metric]


def _detect_unstable_run(summary_row: dict, validation_warnings: int) -> bool:
    values = [
        float(summary_row[column])
        for column in FINAL_METRIC_COLUMNS
        if column in summary_row and pd.notna(summary_row[column])
    ]
    if validation_warnings > 0:
        return True
    if not values or not np.isfinite(values).all():
        return True
    return bool(max(abs(value) for value in values) > 1_000_000)


def _all_metric_values_finite(runs: pd.DataFrame) -> bool:
    if runs.empty:
        return False
    columns = [column for column in FINAL_METRIC_COLUMNS if column in runs.columns]
    if not columns:
        return False
    values = runs[columns].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def _half_sample_stability_check(
    runs: pd.DataFrame,
    *,
    relative_shift_threshold: float = 0.35,
) -> dict[str, float | str | bool]:
    """Compare first-half and second-half MC means as a compact stability check."""

    if runs.empty or "iteration" not in runs.columns:
        return {
            "check": "monte_carlo_half_sample_stability",
            "passed": False,
            "detail": "no_runs_available",
        }

    max_relative_shift = 0.0
    comparisons = 0
    for scenario, frame in runs.groupby("scenario"):
        midpoint = frame["iteration"].median()
        first_half = frame.loc[frame["iteration"] <= midpoint]
        second_half = frame.loc[frame["iteration"] > midpoint]
        if first_half.empty or second_half.empty:
            continue
        for metric in FINAL_METRIC_COLUMNS:
            if metric not in frame.columns:
                continue
            first_mean = float(first_half[metric].mean())
            second_mean = float(second_half[metric].mean())
            full_mean = float(frame[metric].mean())
            denominator = max(abs(full_mean), 1e-9)
            relative_shift = abs(second_mean - first_mean) / denominator
            max_relative_shift = max(max_relative_shift, relative_shift)
            comparisons += 1

    if comparisons == 0:
        return {
            "check": "monte_carlo_half_sample_stability",
            "passed": False,
            "detail": "insufficient_half_sample_comparisons",
        }
    return {
        "check": "monte_carlo_half_sample_stability",
        "passed": bool(max_relative_shift <= relative_shift_threshold),
        "detail": (
            f"max_relative_mean_shift={max_relative_shift:.4f};"
            f"threshold={relative_shift_threshold:.4f};comparisons={comparisons}"
        ),
    }
