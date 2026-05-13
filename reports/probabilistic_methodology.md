# Probabilistic Forecasting And Uncertainty Methodology

This document describes the first exploratory probabilistic layer for the
Algorithmic Nation-State project. It combines closed-form Bayesian updates with
Monte Carlo scenario perturbations around the ECAIF ABM. It is not a calibrated
geopolitical prediction system.

## Bayesian Updating

The Bayesian layer uses conjugate Beta-Binomial updates for binary event
evidence and Dirichlet updates for scenario weights. Priors are weak and
symmetric when no calibration source is supplied. The implementation avoids MCMC
for this first version, so convergence diagnostics are not applicable; posterior
consistency is checked by validating the closed-form arithmetic.

The event examples currently include:

- compute concentration increases
- trade fragmentation intensifies
- AI capability divergence widens
- cloud fragmentation increases
- semiconductor bottlenecks persist
- compute shortages persist

Evidence comes from existing ABM tables and forecast comparison tables. These
are evidence proxies from simulations and model diagnostics, not empirical
event-frequency observations.

## Scenario Probability Adjustments

Scenario probability evolution uses a symmetric Dirichlet prior and cumulative
ABM shock counts as stress evidence. This should be interpreted as an
assumption-transparent scenario-weighting exercise, not a measured probability
that a scenario will occur.

## Monte Carlo Uncertainty Propagation

The Monte Carlo layer samples scenario perturbations and reruns the ABM under
the configured scenarios:

- baseline_globalization
- fragmented_mercantilism
- compute_cold_war
- cooperative_equilibrium

Current runtime used `100` iterations and `30` timesteps per
scenario run.

## Uncertainty Sources

- `compute_supply_multiplier`: normal, mean=1.0, range=[0.7, 1.25]. Perturbs compute infrastructure growth; lower values represent supply constraints.
- `capital_access_multiplier`: normal, mean=1.0, range=[0.65, 1.3]. Perturbs capital mobility assumptions in each scenario.
- `energy_constraint_multiplier`: normal, mean=1.0, range=[0.75, 1.15]. Perturbs compute growth through stylized energy availability.
- `export_control_intensity`: normal, mean=1.0, range=[0.6, 1.55]. Perturbs shock probability and intensity for export-control stress.
- `r_and_d_acceleration_multiplier`: normal, mean=1.0, range=[0.75, 1.35]. Perturbs firm R&D reinvestment rates.
- `ai_productivity_multiplier`: normal, mean=1.0, range=[0.7, 1.4]. Perturbs compute demand and productivity assumptions.
- `data_fragmentation_intensity`: normal, mean=1.0, range=[0.6, 1.5]. Perturbs data-sharing and fragmentation pressure.

These ranges are subjective first-version stress-test assumptions. They are not
calibrated to observed AI infrastructure, finance, energy, or semiconductor
supply-chain data.

## Sensitivity Analysis

Sensitivity rankings use Spearman rank correlations between sampled parameters
and final ABM outcomes. This is a transparent screening diagnostic, not a causal
decomposition.

## Validation

- `requested_runs_completed`: passed=True; observed=400;expected=400;failures=0
- `unstable_simulations_detected`: passed=True; unstable_runs=0
- `finite_output_metrics`: passed=True; all_final_metric_values_finite
- `monte_carlo_half_sample_stability`: passed=True; max_relative_mean_shift=0.1429;threshold=0.3500;comparisons=32
- `fixed_seed_monte_carlo_reproducibility`: passed=True; compact_check_identical

## Interpretation Guidance

- Treat posteriors as conditional on weak priors and simulation-derived evidence
  proxies.
- Treat Monte Carlo outputs as uncertainty propagation through stated
  assumptions.
- Do not describe scenario weights as real-world geopolitical probabilities.
- Do not compare simulated economic output proxies to GDP.
- Do not make causal claims from sensitivity correlations.

## Calibration Limitations

- No external AI infrastructure dataset is used for calibration.
- Shock probabilities and intensities remain scenario assumptions.
- Taiwan uses median-based ABM initialization because the current World Bank
  panel has no observed Taiwan rows.
- Annual macroeconomic data are small and noisy for forecast validation.
- Monte Carlo distributions are inspectable stress-test ranges, not empirical
  distributions.

## Future Extensions

- Add optional calibration files for priors and parameter distributions.
- Calibrate compute and energy constraints using external infrastructure data.
- Add posterior predictive checks once calibrated likelihoods exist.
- Compare Monte Carlo sensitivity rankings across alternative ABM mechanisms.
