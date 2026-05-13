"""Agent definitions for the Evolutionary Cloud AI Firms ABM."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.abm.resources import ResourceBundle, bounded, safe_divide


@dataclass
class StateAgent:
    """A state actor shaping AI-industrial policy and trade access."""

    id: str
    country: str
    name: str
    resources: ResourceBundle
    industrial_policy: float
    trade_openness: float
    export_restriction: float
    data_localization: float
    capital_control: float
    alliance_group: str
    economic_output: float = 0.0

    def apply_policy_pressure(
        self,
        *,
        export_pressure: float,
        data_pressure: float,
        capital_pressure: float,
    ) -> None:
        """Apply scenario and shock-driven policy pressure."""

        self.export_restriction = bounded(self.export_restriction + export_pressure)
        self.data_localization = bounded(self.data_localization + data_pressure)
        self.capital_control = bounded(self.capital_control + capital_pressure)
        self.trade_openness = bounded(
            self.trade_openness - 0.4 * export_pressure - 0.2 * data_pressure
        )

    def support_multiplier(self) -> float:
        """Return infrastructure and R&D support from industrial policy."""

        return 1.0 + self.industrial_policy

    def as_record(self, scenario: str, timestep: int) -> dict[str, float | str | int]:
        """Serialize state variables for output."""

        return {
            "scenario": scenario,
            "timestep": timestep,
            "state_id": self.id,
            "country": self.country,
            "name": self.name,
            "industrial_policy": self.industrial_policy,
            "trade_openness": self.trade_openness,
            "export_restriction": self.export_restriction,
            "data_localization": self.data_localization,
            "capital_control": self.capital_control,
            "alliance_group": self.alliance_group,
            "economic_output": self.economic_output,
            **self.resources.as_dict(),
        }


@dataclass
class AIFirmAgent:
    """An AI firm adapting through compute acquisition and R&D reinvestment."""

    id: str
    name: str
    home_country: str
    alliance_group: str
    resources: ResourceBundle
    capability: float
    resilience: float
    market_access: float
    last_external_compute_share: float = 0.0
    last_output: float = 0.0
    shock_memory: list[str] = field(default_factory=list)

    def compute_demand(self, scenario_compute_demand: float) -> float:
        """Return desired compute units for this timestep."""

        return max(1.0, self.capability * scenario_compute_demand + 0.08 * self.resources.capital)

    def absorb_compute(self, acquired_compute: float, external_compute: float) -> None:
        """Update compute stock and dependency after allocation."""

        self.resources.compute = 0.72 * self.resources.compute + acquired_compute
        self.last_external_compute_share = safe_divide(external_compute, acquired_compute)

    def adapt_to_shocks(self, shock_severity: float) -> None:
        """Increase resilience modestly after exposure to shocks."""

        if shock_severity > 0:
            self.resilience = bounded(self.resilience + 0.015 * shock_severity, upper=2.0)

    def update_capability(
        self,
        *,
        innovation_efficiency: float,
        market_access: float,
        data_localization_penalty: float,
        capital_control_penalty: float,
    ) -> None:
        """Update capability and capital from transparent production assumptions."""

        data_effect = max(1.0, self.resources.data) ** 0.22
        compute_effect = max(1.0, self.resources.compute) ** 0.28
        talent_effect = max(1.0, self.resources.talent) ** 0.18
        capital_effect = max(1.0, self.resources.capital) ** 0.12
        innovation_gain = (
            innovation_efficiency
            * compute_effect
            * data_effect
            * talent_effect
            * capital_effect
            / 10.0
        )
        effective_market_access = max(
            0.05,
            market_access * (1.0 - data_localization_penalty) * (1.0 - capital_control_penalty),
        )
        self.capability = max(0.0, 0.985 * self.capability + innovation_gain * effective_market_access)
        self.last_output = self.capability * effective_market_access
        reinvestment = 0.18 * self.last_output
        self.resources.capital = max(0.0, 0.98 * self.resources.capital + reinvestment)
        self.resources.data = max(0.0, self.resources.data * (1.0 + 0.015 * effective_market_access))
        self.resources.talent = max(0.0, self.resources.talent * (1.0 + 0.006 * self.resilience))
        self.resources.clamp_nonnegative()

    def dependency_score(self) -> float:
        """Return external compute dependency score."""

        return bounded(self.last_external_compute_share)

    def as_record(self, scenario: str, timestep: int) -> dict[str, float | str | int]:
        """Serialize firm variables for output."""

        return {
            "scenario": scenario,
            "timestep": timestep,
            "firm_id": self.id,
            "name": self.name,
            "home_country": self.home_country,
            "alliance_group": self.alliance_group,
            "capability": self.capability,
            "resilience": self.resilience,
            "market_access": self.market_access,
            "external_compute_share": self.last_external_compute_share,
            "economic_output": self.last_output,
            **self.resources.as_dict(),
        }


@dataclass
class ComputeProviderAgent:
    """A compute infrastructure provider constrained by capital and energy."""

    id: str
    name: str
    country: str
    alliance_group: str
    resources: ResourceBundle
    capacity: float
    reliability: float
    export_share: float
    available_compute: float = 0.0

    def reset_available_compute(self, compute_shortage: float) -> None:
        """Reset sellable compute capacity after shortage penalties."""

        shortage_multiplier = max(0.05, 1.0 - compute_shortage)
        energy_limit = max(0.1, self.resources.energy / 100.0)
        self.available_compute = self.capacity * self.reliability * shortage_multiplier * min(1.2, energy_limit)

    def scale_infrastructure(
        self,
        *,
        growth_rate: float,
        capital_mobility: float,
        state_support: float,
        compute_shortage: float,
    ) -> None:
        """Scale capacity using capital, energy, policy support, and shortages."""

        shortage_penalty = max(0.0, 1.0 - 0.5 * compute_shortage)
        investment_effect = growth_rate * capital_mobility * state_support * shortage_penalty
        self.capacity = max(0.0, self.capacity * (1.0 + investment_effect))
        self.resources.capital = max(0.0, self.resources.capital * (1.0 + 0.04 * investment_effect))
        self.resources.energy = max(0.0, self.resources.energy * (1.0 + 0.02 * investment_effect))

    def allocate(self, demand: float) -> float:
        """Allocate compute to a firm and reduce available capacity."""

        allocated = min(max(0.0, demand), self.available_compute)
        self.available_compute -= allocated
        return allocated

    def as_record(self, scenario: str, timestep: int) -> dict[str, float | str | int]:
        """Serialize provider variables for output."""

        return {
            "scenario": scenario,
            "timestep": timestep,
            "provider_id": self.id,
            "name": self.name,
            "country": self.country,
            "alliance_group": self.alliance_group,
            "capacity": self.capacity,
            "available_compute": self.available_compute,
            "reliability": self.reliability,
            "export_share": self.export_share,
            **self.resources.as_dict(),
        }
