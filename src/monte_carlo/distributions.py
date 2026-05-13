"""Parameter distributions for exploratory Monte Carlo uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParameterDistribution:
    """A simple bounded distribution for one uncertain scenario parameter."""

    name: str
    distribution: str
    mean: float
    lower: float
    upper: float
    std: float | None = None
    notes: str = ""

    def sample(self, rng: np.random.Generator) -> float:
        """Sample one value using the configured distribution."""

        if self.distribution == "normal":
            if self.std is None:
                raise ValueError(f"Normal distribution {self.name} requires std.")
            value = rng.normal(self.mean, self.std)
        elif self.distribution == "uniform":
            value = rng.uniform(self.lower, self.upper)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        return float(np.clip(value, self.lower, self.upper))


def default_uncertainty_distributions() -> tuple[ParameterDistribution, ...]:
    """Return inspectable first-version uncertainty distributions.

    These are not calibrated empirical distributions. They are scenario
    perturbation ranges used to test sensitivity to transparent assumptions.
    """

    return (
        ParameterDistribution(
            name="compute_supply_multiplier",
            distribution="normal",
            mean=1.0,
            std=0.12,
            lower=0.70,
            upper=1.25,
            notes="Perturbs compute infrastructure growth; lower values represent supply constraints.",
        ),
        ParameterDistribution(
            name="capital_access_multiplier",
            distribution="normal",
            mean=1.0,
            std=0.14,
            lower=0.65,
            upper=1.30,
            notes="Perturbs capital mobility assumptions in each scenario.",
        ),
        ParameterDistribution(
            name="energy_constraint_multiplier",
            distribution="normal",
            mean=1.0,
            std=0.10,
            lower=0.75,
            upper=1.15,
            notes="Perturbs compute growth through stylized energy availability.",
        ),
        ParameterDistribution(
            name="export_control_intensity",
            distribution="normal",
            mean=1.0,
            std=0.18,
            lower=0.60,
            upper=1.55,
            notes="Perturbs shock probability and intensity for export-control stress.",
        ),
        ParameterDistribution(
            name="r_and_d_acceleration_multiplier",
            distribution="normal",
            mean=1.0,
            std=0.12,
            lower=0.75,
            upper=1.35,
            notes="Perturbs firm R&D reinvestment rates.",
        ),
        ParameterDistribution(
            name="ai_productivity_multiplier",
            distribution="normal",
            mean=1.0,
            std=0.15,
            lower=0.70,
            upper=1.40,
            notes="Perturbs compute demand and productivity assumptions.",
        ),
        ParameterDistribution(
            name="data_fragmentation_intensity",
            distribution="normal",
            mean=1.0,
            std=0.16,
            lower=0.60,
            upper=1.50,
            notes="Perturbs data-sharing and fragmentation pressure.",
        ),
    )


def sample_parameters(
    distributions: Iterable[ParameterDistribution],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample one Monte Carlo parameter set."""

    return {distribution.name: distribution.sample(rng) for distribution in distributions}


def distributions_to_frame(
    distributions: Iterable[ParameterDistribution],
) -> pd.DataFrame:
    """Serialize parameter distributions for reproducibility documentation."""

    return pd.DataFrame([distribution.__dict__ for distribution in distributions])
