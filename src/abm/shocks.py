"""Geopolitical shock generation for the ECAIF ABM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShockEvent:
    """A shock event active during one timestep."""

    timestep: int
    shock_type: str
    severity: float
    affected_countries: tuple[str, ...]
    description: str


@dataclass
class ShockState:
    """Aggregated shock intensities for a timestep."""

    tariffs: float = 0.0
    export_restrictions: float = 0.0
    cloud_fragmentation: float = 0.0
    data_localization: float = 0.0
    capital_controls: float = 0.0
    compute_shortage: float = 0.0

    def total_severity(self) -> float:
        """Return average active shock severity."""

        values = [
            self.tariffs,
            self.export_restrictions,
            self.cloud_fragmentation,
            self.data_localization,
            self.capital_controls,
            self.compute_shortage,
        ]
        return float(sum(values) / len(values))


def generate_shocks(
    *,
    timestep: int,
    countries: list[str],
    rng: np.random.Generator,
    scenario_name: str,
    shock_probability: float,
    shock_intensity: float,
    targeted_fragmentation: bool,
) -> list[ShockEvent]:
    """Generate seeded geopolitical shocks for a timestep."""

    shock_types = [
        "tariffs",
        "semiconductor_export_restrictions",
        "cloud_fragmentation",
        "data_localization",
        "capital_controls",
        "compute_shortages",
    ]
    events: list[ShockEvent] = []
    for shock_type in shock_types:
        if rng.random() > shock_probability:
            continue
        severity = float(np.clip(rng.normal(shock_intensity, 0.08), 0.05, 0.95))
        if targeted_fragmentation:
            affected_count = max(1, int(np.ceil(len(countries) * 0.45)))
            affected = tuple(rng.choice(countries, size=affected_count, replace=False).tolist())
        else:
            affected = tuple(countries)
        events.append(
            ShockEvent(
                timestep=timestep,
                shock_type=shock_type,
                severity=severity,
                affected_countries=affected,
                description=f"{scenario_name} {shock_type} shock",
            )
        )
    return events


def aggregate_shocks(events: list[ShockEvent]) -> ShockState:
    """Aggregate events into timestep modifiers."""

    state = ShockState()
    for event in events:
        if event.shock_type == "tariffs":
            state.tariffs = max(state.tariffs, event.severity)
        elif event.shock_type == "semiconductor_export_restrictions":
            state.export_restrictions = max(state.export_restrictions, event.severity)
        elif event.shock_type == "cloud_fragmentation":
            state.cloud_fragmentation = max(state.cloud_fragmentation, event.severity)
        elif event.shock_type == "data_localization":
            state.data_localization = max(state.data_localization, event.severity)
        elif event.shock_type == "capital_controls":
            state.capital_controls = max(state.capital_controls, event.severity)
        elif event.shock_type == "compute_shortages":
            state.compute_shortage = max(state.compute_shortage, event.severity)
    return state
