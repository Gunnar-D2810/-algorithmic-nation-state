# Algorithmic Nation-State

Algorithmic Nation-State is a reproducible research framework for studying
modern mercantilism, evolutionary cloud AI firms, macroeconomic forecasting,
agent-based simulation, Bayesian updating, and Monte Carlo sensitivity analysis.

## Macroeconomic Data Ingestion

The first implemented pipeline fetches World Bank indicators configured in
`config/indicators.yaml`.

Run from the repository root:

```bash
python scripts/fetch_world_bank_data.py
```

Optional arguments:

```bash
python scripts/fetch_world_bank_data.py \
  --config config/indicators.yaml \
  --project-root .
```

The pipeline:

- reads countries and World Bank indicator mappings from `config/indicators.yaml`
- fetches each configured country/indicator pair from the World Bank API
- saves raw JSON responses with request metadata to `data/raw/world_bank/`
- writes the cleaned long panel to `data/processed/macro_panel.csv`
- writes missingness diagnostics to:
  - `data/processed/macro_panel_missing_report.csv`
  - `data/processed/missing_indicators_by_country.json`

The cleaned panel contains the required columns:

- `country`
- `year`
- `indicator`
- `value`

Additional metadata columns are preserved, including source indicator code,
description, unit, frequency, transformation label, source, and retrieval time.

Missing values are not filled. If a configured indicator is unavailable for a
country, the pipeline reports it rather than inventing or imputing data.

## Baseline Forecasting

The first forecasting pipeline compares transparent baseline models for
`GDP_GROWTH`:

- ARIMA
- SARIMAX
- RandomForest

Run from the repository root after generating `data/processed/macro_panel.csv`:

```bash
python scripts/run_forecasting.py
```

If the project virtual environment is not active, use:

```bash
.venv/bin/python scripts/run_forecasting.py
```

The forecasting pipeline uses expanding-window time-series validation. Rows are
never shuffled, and each test fold occurs after its training fold. For SARIMAX
and RandomForest, predictor variables are lagged by one year so year `t` GDP
growth is predicted from information observed by year `t-1`.

Outputs are written to:

- `reports/tables/forecast_predictions.csv`
- `reports/tables/forecast_fold_metrics.csv`
- `reports/tables/forecast_model_comparison.csv`
- `reports/tables/forecast_failed_country_runs.csv`
- `reports/figures/forecast_actual_vs_predicted_<country>_<model>.png`
- `reports/figures/forecast_residuals_<country>_<model>.png`
- `reports/figures/forecast_model_comparison_rmse.png`

These are baseline models only. They are intended to establish a reproducible
comparison workflow, not to claim strong predictive power. Missing values are
not imputed; country/model combinations with insufficient observed data are
logged in the failed-runs table.

## Modern Forecasting Benchmarks

The second forecasting layer adds exploratory modern benchmarks while preserving
the baseline workflow:

- Prophet
- LightGBM
- LSTM
- optional NeuralForecast models: NBEATS, NHITS, and TFT/PatchTST

Run from the repository root:

```bash
.venv/bin/python scripts/run_modern_forecasting.py
```

You can select a subset:

```bash
.venv/bin/python scripts/run_modern_forecasting.py --models lightgbm,lstm
```

The runner always includes a naive last-observed-value benchmark so modern
models can be checked against a simple reference. Outputs are written to:

- `reports/tables/modern_forecast_predictions.csv`
- `reports/tables/modern_forecast_fold_metrics.csv`
- `reports/tables/modern_forecast_comparison.csv`
- `reports/tables/all_model_comparison.csv`
- `reports/tables/modern_forecast_failed_model_runs.csv`
- `reports/tables/modern_forecast_naive_warnings.csv`
- `reports/figures/modern_model_actual_vs_predicted/`
- `reports/figures/model_rankings.png`

Interpretation guidance:

- These models are exploratory benchmarks on annual macroeconomic data.
- LightGBM can run with the current dependency stack, but it still has few
  country-level training examples.
- Prophet, LSTM, and NeuralForecast models are optional. If their dependencies
  are unavailable or the sample size is too small, the run logs them as skipped
  rather than producing placeholder results.
- Neural and transformer-style forecasts should not be treated as
  state-of-the-art evidence unless the panel is expanded substantially.
- Missing source values are not imputed; models that need complete lagged
  features use only complete rows.

## Forecast Interpretability

The interpretability layer analyzes lagged macroeconomic predictors for the
tree-based forecasting models:

- RandomForest
- LightGBM
- XGBoost, when the installed environment provides it

Run from the repository root:

```bash
.venv/bin/python scripts/run_interpretability.py
```

The workflow computes permutation importance, tree-native feature importance,
LightGBM/XGBoost native contribution summaries, lagged correlations, and a
small rolling-importance diagnostic where there are enough complete rows.

Outputs are written to:

- `reports/tables/feature_importance_global.csv`
- `reports/tables/feature_importance_by_country.csv`
- `reports/tables/correlation_matrix.csv`
- `reports/tables/lag_relationships.csv`
- `reports/tables/rolling_feature_importance.csv`
- `reports/tables/shap_summary.csv`
- `reports/figures/global_feature_importance.png`
- `reports/figures/country_feature_importance/`
- `reports/figures/shap_summary/`
- `reports/figures/correlation_heatmap.png`
- `reports/interpretability_summary.md`

Interpretability results are predictive diagnostics, not causal evidence.
Signals should be read conservatively because the annual country-level sample is
small after lagging and complete-case filtering.

## ECAIF Agent-Based Simulation

The first Evolutionary Cloud AI Firms ABM simulates exploratory interactions
among states, AI firms, and compute providers under four scenarios:

- `baseline_globalization`
- `fragmented_mercantilism`
- `compute_cold_war`
- `cooperative_equilibrium`

Run from the repository root:

```bash
.venv/bin/python scripts/run_abm.py
```

Outputs are written to:

- `reports/tables/abm/`
- `reports/figures/abm/`
- `reports/abm_methodology.md`

The ABM uses stylized resource indexes for compute, data, capital, energy, and
talent. It is a conceptual simulation framework, not a geopolitical prediction
engine or calibrated empirical model.
