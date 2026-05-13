"""Posterior update utilities for the probabilistic forecasting layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

from src.bayesian.likelihoods import EvidenceRecord
from src.bayesian.priors import BetaPrior, DirichletPrior


@dataclass(frozen=True)
class BetaPosterior:
    """Closed-form Beta posterior after binary evidence counts."""

    event_name: str
    scope_type: str
    scope_value: str
    prior_alpha: float
    prior_beta: float
    successes: float
    trials: float
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    credible_interval_lower: float
    credible_interval_upper: float
    evidence_source: str
    notes: str
    prior_source: str


def update_beta_prior(
    prior: BetaPrior,
    evidence: EvidenceRecord,
    *,
    interval_mass: float = 0.9,
) -> BetaPosterior:
    """Update a Beta prior with success/trial evidence."""

    if evidence.trials < evidence.successes:
        raise ValueError("Evidence successes cannot exceed trials.")
    posterior_alpha = prior.alpha + evidence.successes
    posterior_beta = prior.beta + evidence.trials - evidence.successes
    lower_tail = (1.0 - interval_mass) / 2.0
    upper_tail = 1.0 - lower_tail
    total = posterior_alpha + posterior_beta
    return BetaPosterior(
        event_name=evidence.event_name,
        scope_type=evidence.scope_type,
        scope_value=evidence.scope_value,
        prior_alpha=prior.alpha,
        prior_beta=prior.beta,
        successes=evidence.successes,
        trials=evidence.trials,
        posterior_alpha=posterior_alpha,
        posterior_beta=posterior_beta,
        posterior_mean=posterior_alpha / total,
        credible_interval_lower=float(beta_distribution.ppf(lower_tail, posterior_alpha, posterior_beta)),
        credible_interval_upper=float(beta_distribution.ppf(upper_tail, posterior_alpha, posterior_beta)),
        evidence_source=evidence.evidence_source,
        notes=evidence.notes,
        prior_source=prior.source,
    )


def posterior_dataframe(posteriors: list[BetaPosterior]) -> pd.DataFrame:
    """Convert posterior records to a stable CSV-ready dataframe."""

    rows = [posterior.__dict__ for posterior in posteriors]
    columns = [
        "event_name",
        "scope_type",
        "scope_value",
        "prior_alpha",
        "prior_beta",
        "successes",
        "trials",
        "posterior_alpha",
        "posterior_beta",
        "posterior_mean",
        "credible_interval_lower",
        "credible_interval_upper",
        "evidence_source",
        "prior_source",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def update_dirichlet_prior(
    prior: DirichletPrior,
    evidence_counts: dict[str, float],
) -> pd.DataFrame:
    """Update a Dirichlet prior with nonnegative scenario evidence counts."""

    prior_map = prior.as_mapping()
    rows: list[dict[str, float | str]] = []
    posterior_values: list[float] = []
    for category in prior.categories:
        evidence = float(evidence_counts.get(category, 0.0))
        if evidence < 0:
            raise ValueError("Dirichlet evidence counts must be nonnegative.")
        posterior_alpha = prior_map[category] + evidence
        posterior_values.append(posterior_alpha)
        rows.append(
            {
                "scenario": category,
                "prior_alpha": float(prior_map[category]),
                "evidence_count": evidence,
                "posterior_alpha": posterior_alpha,
                "prior_source": prior.source,
                "notes": prior.notes,
            }
        )

    total = float(np.sum(posterior_values))
    for row, posterior_alpha in zip(rows, posterior_values):
        row["posterior_probability"] = posterior_alpha / total if total else float("nan")
    return pd.DataFrame(rows)


def validate_probability_normalization(
    probabilities: pd.DataFrame,
    *,
    group_columns: list[str],
    probability_column: str = "posterior_probability",
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return diagnostics for probability sums by group."""

    if probabilities.empty:
        return pd.DataFrame(columns=[*group_columns, "probability_sum", "passed"])

    sums = (
        probabilities.groupby(group_columns, dropna=False)[probability_column]
        .sum()
        .reset_index(name="probability_sum")
    )
    sums["passed"] = np.isclose(sums["probability_sum"], 1.0, atol=tolerance)
    return sums


def validate_posterior_consistency(posteriors: pd.DataFrame) -> pd.DataFrame:
    """Check closed-form Beta posterior arithmetic."""

    if posteriors.empty:
        return pd.DataFrame(
            [{"check": "posterior_consistency", "passed": False, "detail": "no_posteriors"}]
        )

    expected_alpha = posteriors["prior_alpha"] + posteriors["successes"]
    expected_beta = posteriors["prior_beta"] + posteriors["trials"] - posteriors["successes"]
    alpha_ok = np.allclose(posteriors["posterior_alpha"], expected_alpha)
    beta_ok = np.allclose(posteriors["posterior_beta"], expected_beta)
    bounds_ok = (
        (posteriors["posterior_mean"] >= 0.0)
        & (posteriors["posterior_mean"] <= 1.0)
        & (posteriors["credible_interval_lower"] >= 0.0)
        & (posteriors["credible_interval_upper"] <= 1.0)
    ).all()
    return pd.DataFrame(
        [
            {
                "check": "posterior_consistency",
                "passed": bool(alpha_ok and beta_ok and bounds_ok),
                "detail": (
                    "closed_form_beta_updates_valid"
                    if alpha_ok and beta_ok and bounds_ok
                    else "posterior_arithmetic_or_bounds_failed"
                ),
            }
        ]
    )
