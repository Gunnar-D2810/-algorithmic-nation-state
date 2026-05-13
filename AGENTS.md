# AGENTS.md

## Project

Algorithmic Nation-State is a research framework exploring:
- modern mercantilism
- evolutionary AI firms (ECAIF)
- macroeconomic forecasting
- agent-based modeling
- Bayesian inference
- Monte Carlo simulation
- AI infrastructure and geopolitical dynamics

Primary goal:
Build reproducible forecasting and simulation systems for human-AI economic systems.

---

## Tech Stack

- Python 3.12
- pandas
- numpy
- scipy
- scikit-learn
- statsmodels
- matplotlib
- networkx
- pymc
- jupyter

---

## Repository Structure

src/data
- data ingestion
- preprocessing
- API connectors

src/models
- forecasting models
- ARIMA
- RandomForest
- XGBoost

src/abm
- agent-based models
- ECAIF simulations
- equilibrium dynamics

src/bayesian
- Bayesian updates
- probabilistic forecasts

src/monte_carlo
- sensitivity analysis
- scenario simulation

reports
- generated markdown reports
- figures
- tables

papers
- conceptual papers
- appendices
- methodology notes

---

## Coding Standards

- Write modular Python code
- Prefer small reusable functions
- Use type hints where practical
- Add docstrings to major functions
- Avoid hardcoded paths
- Save outputs reproducibly

---

## Research Standards

- Do not fabricate empirical results
- Clearly separate:
  - conceptual assumptions
  - simulated outputs
  - empirical findings

- Mark placeholder data clearly
- Document all assumptions
- Keep methods reproducible

---

## Visualization Standards

- Use matplotlib or plotly
- Save figures to reports/figures
- Use clear academic chart labels
- No excessive styling

---

## Forecasting Standards

Compare:
- ARIMA / SARIMAX
- RandomForest
- optional gradient boosting

Metrics:
- RMSE
- MAE
- MAPE where applicable

Use:
- time-series split validation
- out-of-sample evaluation

---

## ABM Standards

Agents:
- states
- AI firms
- compute providers

Resources:
- compute
- data
- capital
- energy

Shocks:
- tariffs
- export controls
- data localization
- hardware bottlenecks

---

## Done Criteria

Tasks are complete when:
- code runs end-to-end
- outputs are reproducible
- figures are generated
- notebooks execute cleanly
- assumptions are documented

---

## Constraints

Never:
- invent citations
- fake quantitative validation
- overwrite user work without explicit request

Always:
- explain assumptions
- preserve reproducibility
- prefer transparent implementations
