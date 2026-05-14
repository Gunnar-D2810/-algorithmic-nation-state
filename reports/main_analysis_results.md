# Main Analysis Results

This report integrates the current reproducible outputs for the Algorithmic
Nation-State project. Findings are intentionally conservative: empirical
forecasting diagnostics, exploratory simulations, and subjective probabilistic
updates are separated rather than blended into a single prediction.

## Scope

- Forecast target: `GDP_GROWTH`
- Countries: `USA, DEU, CHN, JPN, IND, IDN, TWN`
- Integrated figures: `/Users/Gunnar/Documents/algorithmic-nation-state/reports/figures/integrated`

## Empirical Data Coverage

| country | rows | observed_values |
| --- | --- | --- |
| CHN | 594 | 432 |
| DEU | 594 | 476 |
| IDN | 594 | 473 |
| IND | 594 | 495 |
| JPN | 594 | 477 |
| TWN | 0 | 0 |
| USA | 594 | 477 |

## Forecasting Benchmarks

| model | mean_rmse | mean_mae | countries |
| --- | --- | --- | --- |
| ARIMA | 2.7448 | 1.8346 | 6 |
| naive | 3.0070 | 2.1753 | 6 |
| RandomForest | 3.1119 | 2.3395 | 6 |
| lightgbm | 3.2294 | 2.3958 | 6 |
| SARIMAX | 7.1683 | 5.8450 | 6 |

These rows summarize observed model errors from the generated comparison table.
They should not be read as evidence of structural causal relationships.

## Forecast Interpretability

| model | feature_base | feature_group | mean_permutation_importance_mean | countries | stability_warning |
| --- | --- | --- | --- | --- | --- |
| xgboost | INFLATION | traditional_macro | 0.0939 | 6 | importance varies substantially across countries |
| xgboost | UNEMPLOYMENT | traditional_macro | 0.0543 | 6 | importance varies substantially across countries |
| lightgbm | FDI_NET_INFLOWS | capital_flows | 0.0238 | 6 | importance varies substantially across countries |
| lightgbm | UNEMPLOYMENT | traditional_macro | 0.0237 | 6 | importance varies substantially across countries |
| xgboost | MILITARY_EXPENDITURE | military_expenditure | 0.0098 | 6 | importance varies substantially across countries |
| xgboost | HIGH_TECH_EXPORTS | technology_trade | 0.0040 | 6 | importance varies substantially across countries |
| random_forest | MILITARY_EXPENDITURE | military_expenditure | 0.0017 | 6 | importance varies substantially across countries |
| random_forest | UNEMPLOYMENT | traditional_macro | 0.0012 | 6 | importance varies substantially across countries |
| lightgbm | MILITARY_EXPENDITURE | military_expenditure | 0.0000 | 6 | importance varies substantially across countries |
| xgboost | EXPORTS_PERCENT_GDP | trade | 0.0000 | 6 | importance varies substantially across countries |

Feature importance is predictive and model-dependent. It is not causal evidence.

## ECAIF ABM Scenario Outputs

| scenario | final_economic_output_proxy | final_dependency_score | final_ai_capability_divergence | total_shock_events |
| --- | --- | --- | --- | --- |
| baseline_globalization | 44.3824 | 0.4343 | 0.3398 | 11.0000 |
| fragmented_mercantilism | 14.0816 | 0.0973 | 0.2503 | 33.0000 |
| compute_cold_war | 9.3764 | 0.0183 | 0.2413 | 44.0000 |
| cooperative_equilibrium | 53.2806 | 0.5117 | 0.3916 | 12.0000 |

ABM values are simulated diagnostics conditional on stated assumptions.

## Bayesian Update Outputs

| event_name | scope_type | scope_value | posterior_mean | credible_interval_lower | credible_interval_upper |
| --- | --- | --- | --- | --- | --- |
| compute_concentration_increases | scenario | baseline_globalization | 0.7500 | 0.6174 | 0.8646 |
| ai_capability_divergence_widens | scenario | baseline_globalization | 0.8750 | 0.7685 | 0.9547 |
| cloud_fragmentation_increases | scenario | baseline_globalization | 0.1250 | 0.0453 | 0.2315 |
| semiconductor_bottlenecks_persist | scenario | baseline_globalization | 0.1875 | 0.0878 | 0.3096 |
| compute_shortage_persists | scenario | baseline_globalization | 0.1250 | 0.0453 | 0.2315 |
| trade_fragmentation_intensifies | scenario | baseline_globalization | 0.2812 | 0.1606 | 0.4177 |
| compute_concentration_increases | scenario | compute_cold_war | 0.6250 | 0.4818 | 0.7592 |
| ai_capability_divergence_widens | scenario | compute_cold_war | 0.9688 | 0.9079 | 0.9983 |
| cloud_fragmentation_increases | scenario | compute_cold_war | 0.2812 | 0.1606 | 0.4177 |
| semiconductor_bottlenecks_persist | scenario | compute_cold_war | 0.3750 | 0.2408 | 0.5182 |
| compute_shortage_persists | scenario | compute_cold_war | 0.2812 | 0.1606 | 0.4177 |
| trade_fragmentation_intensifies | scenario | compute_cold_war | 0.5938 | 0.4496 | 0.7312 |

Posterior values use weak priors and simulation/model evidence proxies. They are
not calibrated real-world geopolitical probabilities.

## Monte Carlo Summary

| scenario | metric | mean | q05 | q95 | unstable_run_rate |
| --- | --- | --- | --- | --- | --- |
| baseline_globalization | final_ai_capability_divergence | 0.3547 | 0.2858 | 0.4265 | 0.0000 |
| baseline_globalization | final_dependency_score | 0.4148 | 0.3750 | 0.4381 | 0.0000 |
| baseline_globalization | final_economic_output_proxy | 42.6621 | 33.3021 | 51.5375 | 0.0000 |
| compute_cold_war | final_ai_capability_divergence | 0.2344 | 0.2003 | 0.2739 | 0.0000 |
| compute_cold_war | final_dependency_score | 0.0152 | 0.0068 | 0.0197 | 0.0000 |
| compute_cold_war | final_economic_output_proxy | 10.4457 | 7.2791 | 12.9376 | 0.0000 |
| cooperative_equilibrium | final_ai_capability_divergence | 0.3652 | 0.2925 | 0.4360 | 0.0000 |
| cooperative_equilibrium | final_dependency_score | 0.5172 | 0.4858 | 0.5324 | 0.0000 |
| cooperative_equilibrium | final_economic_output_proxy | 51.6101 | 41.2081 | 61.6771 | 0.0000 |
| fragmented_mercantilism | final_ai_capability_divergence | 0.2633 | 0.2148 | 0.3144 | 0.0000 |
| fragmented_mercantilism | final_dependency_score | 0.1037 | 0.0664 | 0.1269 | 0.0000 |
| fragmented_mercantilism | final_economic_output_proxy | 17.7367 | 13.4508 | 21.4615 | 0.0000 |

## Major Sensitivity Drivers

| parameter | mean_absolute_correlation |
| --- | --- |
| ai_productivity_multiplier | 0.3655 |
| r_and_d_acceleration_multiplier | 0.3245 |
| export_control_intensity | 0.2846 |
| data_fragmentation_intensity | 0.1375 |
| capital_access_multiplier | 0.0993 |
| compute_supply_multiplier | 0.0914 |
| energy_constraint_multiplier | 0.0606 |

Sensitivity rankings are rank correlations over sampled ABM perturbations, not
causal decompositions.
