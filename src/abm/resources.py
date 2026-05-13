"""Resource containers and inequality metrics for the ECAIF ABM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ResourceBundle:
    """Core resources used by states, AI firms, and compute providers."""

    compute: float
    data: float
    capital: float
    energy: float
    talent: float

    def as_dict(self) -> dict[str, float]:
        """Return resources as a serializable dictionary."""

        return {
            "compute": self.compute,
            "data": self.data,
            "capital": self.capital,
            "energy": self.energy,
            "talent": self.talent,
        }

    def clamp_nonnegative(self) -> None:
        """Ensure no resource stock becomes negative."""

        self.compute = max(0.0, self.compute)
        self.data = max(0.0, self.data)
        self.capital = max(0.0, self.capital)
        self.energy = max(0.0, self.energy)
        self.talent = max(0.0, self.talent)

    def is_valid(self) -> bool:
        """Check whether resources are finite and nonnegative."""

        values = self.as_dict().values()
        return all(math.isfinite(value) and value >= 0 for value in values)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with a safe fallback for zero denominators."""

    if denominator == 0:
        return default
    return numerator / denominator


def gini(values: list[float] | np.ndarray) -> float:
    """Compute a Gini coefficient for nonnegative values."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0
    array = np.clip(array, 0, None)
    total = array.sum()
    if total == 0:
        return 0.0
    sorted_values = np.sort(array)
    n = len(sorted_values)
    cumulative = np.cumsum(sorted_values)
    return float((n + 1 - 2 * np.sum(cumulative) / total) / n)


def herfindahl_index(values: list[float] | np.ndarray) -> float:
    """Compute the Herfindahl-Hirschman concentration index."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    array = np.clip(array, 0, None)
    total = array.sum()
    if total == 0:
        return 0.0
    shares = array / total
    return float(np.sum(shares**2))


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a scalar to a closed interval."""

    return min(upper, max(lower, value))


def normalized(value: float, low: float, high: float, default: float = 0.5) -> float:
    """Normalize a value to [0, 1] using observed low/high bounds."""

    if not math.isfinite(value) or high <= low:
        return default
    return bounded((value - low) / (high - low))
