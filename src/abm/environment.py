"""Simulation environment for ECAIF agent interactions."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.abm.agents import AIFirmAgent, ComputeProviderAgent, StateAgent
from src.abm.config import ABMConfig, CountryInitialization, ScenarioConfig
from src.abm.metrics import compute_metrics
from src.abm.resources import ResourceBundle, bounded
from src.abm.shocks import ShockEvent, aggregate_shocks, generate_shocks


@dataclass
class SimulationEnvironment:
    """Discrete-timestep environment coordinating states, firms, providers, and shocks."""

    config: ABMConfig
    scenario: ScenarioConfig
    rng: np.random.Generator
    states: list[StateAgent] = field(default_factory=list)
    firms: list[AIFirmAgent] = field(default_factory=list)
    providers: list[ComputeProviderAgent] = field(default_factory=list)
    timestep: int = 0
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    shock_events: list[ShockEvent] = field(default_factory=list)
    network_edge_records: list[dict[str, float | str | int]] = field(default_factory=list)
    validation_records: list[dict[str, float | str | int]] = field(default_factory=list)

    def initialize(self) -> None:
        """Create state, firm, and compute-provider agents."""

        for country in self.config.countries:
            self.states.append(self._create_state(country))
            for provider_index in range(self.config.providers_per_country):
                self.providers.append(self._create_provider(country, provider_index))
            for firm_index in range(self.config.firms_per_country):
                self.firms.append(self._create_firm(country, firm_index))
        self._refresh_graph()

    def step(self) -> dict[str, float | int | str]:
        """Advance the simulation one timestep and return system metrics."""

        self.timestep += 1
        country_codes = [state.country for state in self.states]
        events = generate_shocks(
            timestep=self.timestep,
            countries=country_codes,
            rng=self.rng,
            scenario_name=self.scenario.name,
            shock_probability=self.scenario.shock_probability,
            shock_intensity=self.scenario.shock_intensity,
            targeted_fragmentation=self.scenario.targeted_fragmentation,
        )
        self.shock_events.extend(events)
        shock_state = aggregate_shocks(events)

        self._update_state_policies(events)
        for provider in self.providers:
            state = self._state_for_country(provider.country)
            provider.scale_infrastructure(
                growth_rate=self.scenario.compute_growth_rate,
                capital_mobility=self.scenario.capital_mobility,
                state_support=state.support_multiplier(),
                compute_shortage=shock_state.compute_shortage,
            )
            provider.reset_available_compute(shock_state.compute_shortage)

        for firm in self.firms:
            acquired, external = self._allocate_compute_to_firm(firm, shock_state)
            firm.absorb_compute(acquired, external)
            home_state = self._state_for_country(firm.home_country)
            firm.adapt_to_shocks(shock_state.total_severity())
            firm.update_capability(
                innovation_efficiency=self.scenario.r_and_d_reinvestment,
                market_access=self._market_access(firm, home_state, shock_state),
                data_localization_penalty=home_state.data_localization * 0.45,
                capital_control_penalty=home_state.capital_control * 0.35,
            )

        self._update_state_outputs()
        self._refresh_graph()
        self._validate_state()
        return compute_metrics(
            scenario=self.scenario.name,
            timestep=self.timestep,
            states=self.states,
            firms=self.firms,
            providers=self.providers,
            shock_count=len(events),
        )

    def snapshot_records(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Return serialized state, firm, and provider records."""

        state_records = [
            state.as_record(self.scenario.name, self.timestep) for state in self.states
        ]
        firm_records = [
            firm.as_record(self.scenario.name, self.timestep) for firm in self.firms
        ]
        provider_records = [
            provider.as_record(self.scenario.name, self.timestep)
            for provider in self.providers
        ]
        return state_records, firm_records, provider_records

    def shock_records(self) -> list[dict[str, float | str | int]]:
        """Return serialized shock records."""

        return [
            {
                "scenario": self.scenario.name,
                "timestep": event.timestep,
                "shock_type": event.shock_type,
                "severity": event.severity,
                "affected_countries": ";".join(event.affected_countries),
                "description": event.description,
            }
            for event in self.shock_events
        ]

    def _create_state(self, country: CountryInitialization) -> StateAgent:
        trade_openness = bounded(
            country.trade_openness * self.scenario.trade_openness_multiplier
        )
        base_restriction = bounded((1.0 - trade_openness) * 0.22)
        return StateAgent(
            id=f"state_{country.code}",
            country=country.code,
            name=country.name,
            resources=ResourceBundle(
                compute=country.compute_base,
                data=country.data_base,
                capital=country.capital_base,
                energy=country.energy_base,
                talent=country.talent_base,
            ),
            industrial_policy=country.industrial_policy,
            trade_openness=trade_openness,
            export_restriction=base_restriction,
            data_localization=bounded((1.0 - self.scenario.data_sharing) * 0.35),
            capital_control=bounded((1.0 - self.scenario.capital_mobility) * 0.30),
            alliance_group=_alliance_group(country.code),
        )

    def _create_provider(
        self,
        country: CountryInitialization,
        provider_index: int,
    ) -> ComputeProviderAgent:
        jitter = float(self.rng.normal(1.0, 0.04))
        capacity = max(5.0, country.compute_base * 1.35 * jitter)
        return ComputeProviderAgent(
            id=f"provider_{country.code}_{provider_index + 1}",
            name=f"{country.code} Compute Provider {provider_index + 1}",
            country=country.code,
            alliance_group=_alliance_group(country.code),
            resources=ResourceBundle(
                compute=capacity,
                data=country.data_base * 0.35,
                capital=country.capital_base * 0.70,
                energy=country.energy_base * 1.20,
                talent=country.talent_base * 0.35,
            ),
            capacity=capacity,
            reliability=bounded(0.78 + 0.16 * self.scenario.cooperation_factor, upper=0.99),
            export_share=bounded(country.trade_openness * self.scenario.trade_openness_multiplier),
        )

    def _create_firm(self, country: CountryInitialization, firm_index: int) -> AIFirmAgent:
        jitter = float(self.rng.normal(1.0, 0.05))
        scale = 0.62 + 0.12 * firm_index
        return AIFirmAgent(
            id=f"firm_{country.code}_{firm_index + 1}",
            name=f"{country.code} AI Firm {firm_index + 1}",
            home_country=country.code,
            alliance_group=_alliance_group(country.code),
            resources=ResourceBundle(
                compute=country.compute_base * scale * jitter,
                data=country.data_base * scale,
                capital=country.capital_base * scale,
                energy=country.energy_base * 0.45,
                talent=country.talent_base * scale,
            ),
            capability=max(1.0, (country.compute_base + country.data_base + country.talent_base) / 28.0 * jitter),
            resilience=bounded(0.45 + 0.25 * self.scenario.cooperation_factor, upper=1.2),
            market_access=bounded(country.trade_openness * self.scenario.trade_openness_multiplier),
        )

    def _allocate_compute_to_firm(self, firm: AIFirmAgent, shock_state) -> tuple[float, float]:
        demand = firm.compute_demand(self.scenario.compute_demand_factor)
        accessible = self._accessible_providers(firm, shock_state)
        acquired = 0.0
        external = 0.0
        weighted_capacity = sum(
            access_weight * provider.available_compute
            for provider, access_weight in accessible
        )
        if weighted_capacity <= 0:
            return acquired, external
        for provider, access_weight in accessible:
            target_allocation = demand * (
                access_weight * provider.available_compute / weighted_capacity
            )
            allocation = provider.allocate(target_allocation)
            acquired += allocation
            if provider.country != firm.home_country:
                external += allocation
        compute_cost = acquired * (0.015 + 0.04 * shock_state.tariffs)
        firm.resources.capital = max(0.0, firm.resources.capital - compute_cost)
        return acquired, external

    def _accessible_providers(self, firm: AIFirmAgent, shock_state) -> list[tuple[ComputeProviderAgent, float]]:
        providers: list[tuple[ComputeProviderAgent, float]] = []
        home_state = self._state_for_country(firm.home_country)
        for provider in self.providers:
            provider_state = self._state_for_country(provider.country)
            same_country = provider.country == firm.home_country
            same_alliance = provider.alliance_group == firm.alliance_group
            if same_country:
                access = 1.0
            elif same_alliance:
                access = (
                    0.70
                    * self.scenario.cooperation_factor
                    * provider.export_share
                    * (1.0 - shock_state.export_restrictions * 0.45)
                )
            else:
                access = (
                    home_state.trade_openness
                    * provider_state.trade_openness
                    * self.scenario.trade_openness_multiplier
                    * (1.0 - shock_state.cloud_fragmentation)
                    * (1.0 - shock_state.export_restrictions)
                )
            if access > 0.03:
                providers.append((provider, max(0.0, access)))
        providers.sort(key=lambda item: item[1] * item[0].available_compute, reverse=True)
        return providers

    def _market_access(self, firm: AIFirmAgent, home_state: StateAgent, shock_state) -> float:
        return bounded(
            firm.market_access
            * (1.0 - 0.40 * shock_state.tariffs)
            * (1.0 - 0.35 * shock_state.cloud_fragmentation)
            * (1.0 - 0.20 * home_state.export_restriction),
            lower=0.05,
            upper=1.25,
        )

    def _update_state_policies(self, events: list[ShockEvent]) -> None:
        affected_by_country: dict[str, list[ShockEvent]] = {state.country: [] for state in self.states}
        for event in events:
            for country in event.affected_countries:
                affected_by_country.setdefault(country, []).append(event)
        for state in self.states:
            events_for_state = affected_by_country.get(state.country, [])
            export_pressure = sum(
                event.severity for event in events_for_state
                if event.shock_type in {"tariffs", "semiconductor_export_restrictions"}
            ) * 0.03
            data_pressure = sum(
                event.severity for event in events_for_state
                if event.shock_type in {"data_localization", "cloud_fragmentation"}
            ) * 0.025
            capital_pressure = sum(
                event.severity for event in events_for_state
                if event.shock_type == "capital_controls"
            ) * 0.03
            state.apply_policy_pressure(
                export_pressure=export_pressure,
                data_pressure=data_pressure,
                capital_pressure=capital_pressure,
            )

    def _update_state_outputs(self) -> None:
        for state in self.states:
            firm_output = sum(
                firm.last_output for firm in self.firms if firm.home_country == state.country
            )
            provider_output = sum(
                provider.capacity * 0.04 for provider in self.providers
                if provider.country == state.country
            )
            state.economic_output = firm_output + provider_output

    def _refresh_graph(self) -> None:
        graph = nx.DiGraph()
        for state in self.states:
            graph.add_node(state.id, kind="state", country=state.country)
        for provider in self.providers:
            graph.add_node(provider.id, kind="provider", country=provider.country)
            graph.add_edge(f"state_{provider.country}", provider.id, relation="hosts", weight=1.0)
        for firm in self.firms:
            graph.add_node(firm.id, kind="firm", country=firm.home_country)
            graph.add_edge(f"state_{firm.home_country}", firm.id, relation="charters", weight=1.0)
            for provider in self.providers:
                if provider.country == firm.home_country:
                    graph.add_edge(provider.id, firm.id, relation="home_compute", weight=1.0)
                elif provider.alliance_group == firm.alliance_group:
                    graph.add_edge(provider.id, firm.id, relation="allied_compute", weight=0.4)
        self.graph = graph

        if self.timestep == self.config.time_steps:
            for source, target, attrs in graph.edges(data=True):
                self.network_edge_records.append(
                    {
                        "scenario": self.scenario.name,
                        "timestep": self.timestep,
                        "source": source,
                        "target": target,
                        "relation": attrs.get("relation", ""),
                        "weight": float(attrs.get("weight", 0.0)),
                    }
                )

    def _validate_state(self) -> None:
        warnings: list[str] = []
        for agent in [*self.states, *self.firms, *self.providers]:
            if not agent.resources.is_valid():
                warnings.append(f"invalid_resources:{getattr(agent, 'id', 'unknown')}")
        max_capability = max((firm.capability for firm in self.firms), default=0.0)
        max_capacity = max((provider.capacity for provider in self.providers), default=0.0)
        if max_capability > 1_000_000 or max_capacity > 1_000_000:
            warnings.append("exploding_values_detected")
        self.validation_records.append(
            {
                "scenario": self.scenario.name,
                "timestep": self.timestep,
                "warning_count": len(warnings),
                "warnings": ";".join(warnings),
                "max_capability": max_capability,
                "max_compute_capacity": max_capacity,
            }
        )

    def _state_for_country(self, country: str) -> StateAgent:
        for state in self.states:
            if state.country == country:
                return state
        raise KeyError(f"No state found for country: {country}")


def _alliance_group(country_code: str) -> str:
    """Assign broad conceptual alliance groups for network access rules."""

    if country_code in {"USA", "DEU", "JPN", "TWN"}:
        return "open_stack"
    if country_code in {"CHN"}:
        return "sovereign_stack"
    return "nonaligned_stack"
