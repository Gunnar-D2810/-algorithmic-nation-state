# ECAIF ABM Methodology

This document describes the first exploratory Evolutionary Cloud AI Firms
(ECAIF) agent-based model. The simulation is a conceptual research tool, not a
geopolitical prediction engine.

## Purpose

The model explores how states, AI firms, and compute providers interact under
modern mercantilist conditions. It is designed to make assumptions inspectable
and to generate scenario dynamics for later Bayesian and Monte Carlo analysis.

## Agents

- `StateAgent`: represents state-level policy pressure, trade openness,
  industrial policy, export restrictions, data localization, and capital
  controls.
- `AIFirmAgent`: represents cloud AI firms that acquire compute, accumulate
  data/capital/talent, reinvest into R&D, and adapt to shocks.
- `ComputeProviderAgent`: represents compute infrastructure providers that
  scale capacity under capital, energy, and shortage constraints.

## Resources

Each agent uses transparent resource stocks:

- compute
- data
- capital
- energy
- talent

The initial resource levels use latest World Bank indicators as rough scale
hints where available. Missing country panels, currently Taiwan, receive
median-based conceptual defaults and are explicitly marked in
`reports/tables/abm/abm_initialization.csv`.

## Scenarios

- `baseline_globalization`: Open trade and moderate cooperation with occasional shocks.
- `fragmented_mercantilism`: Higher policy barriers, fragmented data flows, and reduced capital mobility.
- `compute_cold_war`: Persistent compute rivalry with export controls and semiconductor bottlenecks.
- `cooperative_equilibrium`: High coordination, shared standards, and lower fragmentation pressure.

## Conceptual Mechanisms

- Firms demand compute as a function of capability and capital.
- Providers allocate constrained compute capacity to domestic, allied, and
  foreign firms depending on scenario openness and active shocks.
- Firms update capability through simple production-style assumptions using
  compute, data, capital, and talent.
- States adjust policy restrictions in response to shocks.
- Providers scale infrastructure through capital mobility, state support, and
  compute-shortage penalties.

## Geopolitical Shocks

The model includes seeded stochastic shocks:

- tariffs
- semiconductor export restrictions
- cloud fragmentation
- data localization
- capital controls
- compute shortages

Shock probabilities and intensities are scenario assumptions, not empirical
estimates.

## Metrics

The simulation records:

- compute concentration index
- compute inequality
- AI capability divergence
- economic output proxy
- resilience score
- dependency score
- infrastructure asymmetry

These metrics are simulation diagnostics. They should not be interpreted as
validated empirical measurements.

## Validation

The run records invalid resources, exploding values, and fixed-seed
reproducibility. Current reproducibility check:

- scenario: `baseline_globalization`
- passed: `True`
- detail: `metrics_and_firm_tables_identical`

## Limitations

- The model is not calibrated to observed AI firm data.
- Resource stocks are stylized indexes, not measured physical quantities.
- GDP, trade, high-tech exports, FDI, and military expenditure only inform
  initialization heuristics.
- Network access rules use broad conceptual alliance groups.
- Shocks are generated from scenario assumptions, not forecast probabilities.
- The economic output proxy is not GDP.

## Interpretation Guidance

Use this ABM to compare mechanisms and scenario sensitivities, not to predict
country outcomes. Results should be described as simulated outputs conditional
on stated assumptions.

## Future Extensions

- Bayesian parameter updates for shock probabilities and resource elasticities.
- Monte Carlo sensitivity over scenario assumptions.
- Calibration against external AI infrastructure datasets where available.
- More detailed supply-chain and energy constraints.
