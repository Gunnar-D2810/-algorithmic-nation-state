"""Transparent prior definitions for exploratory Bayesian updates.

The priors in this module are deliberately weak and inspectable. They are not
empirical calibrations, and downstream reports should describe them as
subjective or assumption-driven priors unless a future calibration file
explicitly replaces them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_EVENT_NAMES = (
    "compute_concentration_increases",
    "trade_fragmentation_intensifies",
    "ai_capability_divergence_widens",
    "cloud_fragmentation_increases",
    "semiconductor_bottlenecks_persist",
    "compute_shortage_persists",
)


@dataclass(frozen=True)
class BetaPrior:
    """Beta prior for a binary event probability."""

    name: str
    alpha: float
    beta: float
    source: str
    notes: str

    @property
    def mean(self) -> float:
        """Return the prior mean."""

        total = self.alpha + self.beta
        return self.alpha / total if total else float("nan")


@dataclass(frozen=True)
class DirichletPrior:
    """Dirichlet prior for scenario probability weights."""

    name: str
    categories: tuple[str, ...]
    alpha: tuple[float, ...]
    source: str
    notes: str

    def as_mapping(self) -> dict[str, float]:
        """Return category-to-alpha mapping."""

        return dict(zip(self.categories, self.alpha))


def weak_binary_prior(name: str, *, strength: float = 2.0) -> BetaPrior:
    """Create a weak symmetric Beta prior without encoding a preferred outcome."""

    if strength <= 0:
        raise ValueError("Prior strength must be positive.")
    alpha = strength / 2.0
    return BetaPrior(
        name=name,
        alpha=alpha,
        beta=alpha,
        source="weak_symmetric_subjective_prior",
        notes=(
            "No empirical calibration supplied. This prior encodes uncertainty "
            "rather than a directional geopolitical claim."
        ),
    )


def default_event_priors(
    event_names: Iterable[str] = DEFAULT_EVENT_NAMES,
    *,
    strength: float = 2.0,
) -> dict[str, BetaPrior]:
    """Return weak priors for event names used by the forecast update layer."""

    return {name: weak_binary_prior(name, strength=strength) for name in event_names}


def default_scope_prior(scope_event_name: str, *, strength: float = 2.0) -> BetaPrior:
    """Return a weak prior for model-level or country-level update rows."""

    return weak_binary_prior(scope_event_name, strength=strength)


def default_scenario_prior(
    scenario_names: Iterable[str],
    *,
    strength_per_scenario: float = 1.0,
) -> DirichletPrior:
    """Return a symmetric scenario prior over configured ABM scenarios."""

    categories = tuple(scenario_names)
    if not categories:
        raise ValueError("At least one scenario is required for a Dirichlet prior.")
    if strength_per_scenario <= 0:
        raise ValueError("Scenario prior strength must be positive.")
    return DirichletPrior(
        name="scenario_probability_weights",
        categories=categories,
        alpha=tuple(float(strength_per_scenario) for _ in categories),
        source="symmetric_subjective_scenario_prior",
        notes=(
            "Initial scenario weights are equal because no calibrated scenario "
            "probabilities are available."
        ),
    )
