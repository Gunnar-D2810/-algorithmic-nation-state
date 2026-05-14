# Methodology Summary

## Data

World Bank indicators are loaded from `config/indicators.yaml`, stored as raw
JSON, and cleaned into `data/processed/macro_panel.csv`. Missing values are
reported and are not silently filled.

## Forecasting

GDP growth forecasting uses chronological validation. ARIMA is univariate;
SARIMAX, RandomForest, LightGBM, and other optional benchmarks use lagged
features where appropriate to avoid same-year leakage.

## Interpretability

Feature importance uses tree-model permutation importance, native importance,
correlations, lag diagnostics, and contribution summaries where practical.

## Agent-Based Modeling

The ECAIF ABM models states, AI firms, and compute providers with stylized
resources. Scenarios alter openness, cooperation, shocks, compute growth, data
sharing, capital mobility, and R&D reinvestment.

## Probabilistic Layer

Bayesian updates use closed-form Beta and Dirichlet structures over explicitly
labeled evidence proxies. Monte Carlo analysis samples transparent scenario
perturbations and reruns the ABM.

## Reproducibility

Scripts centralize configuration loading, logging, run metadata, deterministic
seeds, schema checks, and output locations.
