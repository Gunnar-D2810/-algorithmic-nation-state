# Interpretability Summary

This report summarizes exploratory feature-importance analysis for GDP growth forecasting. The analysis uses one-year-lagged macroeconomic indicators only; contemporaneous target values are not used as predictors.

## Data Coverage

| country | complete_lagged_rows | first_complete_year | last_complete_year |
| --- | --- | --- | --- |
| USA | 17 | 2008 | 2024 |
| DEU | 17 | 2008 | 2024 |
| CHN | 17 | 2008 | 2024 |
| JPN | 17 | 2008 | 2024 |
| IND | 15 | 2010 | 2024 |
| IDN | 14 | 2011 | 2024 |
| TWN | 0 |  |  |

## Strongest Predictive Indicators

| model | feature_base | feature_group | mean_permutation_importance_mean | mean_native_importance | countries |
| --- | --- | --- | --- | --- | --- |
| xgboost | INFLATION | traditional_macro | 0.0939 | 0.1359 | 6 |
| xgboost | UNEMPLOYMENT | traditional_macro | 0.0543 | 0.3022 | 6 |
| lightgbm | FDI_NET_INFLOWS | capital_flows | 0.0238 | 45.5000 | 6 |
| lightgbm | UNEMPLOYMENT | traditional_macro | 0.0237 | 29.0000 | 6 |
| xgboost | MILITARY_EXPENDITURE | military_expenditure | 0.0098 | 0.1541 | 6 |
| xgboost | HIGH_TECH_EXPORTS | technology_trade | 0.0040 | 0.0603 | 6 |
| random_forest | MILITARY_EXPENDITURE | military_expenditure | 0.0017 | 0.1106 | 6 |
| random_forest | UNEMPLOYMENT | traditional_macro | 0.0012 | 0.1100 | 6 |

## Stable Signals

No clearly stable importance signals were identified.

## Unstable Or Weak Findings

| model | feature_base | feature_group | mean_permutation_importance_mean | instability_score | stability_warning |
| --- | --- | --- | --- | --- | --- |
| xgboost | INFLATION | traditional_macro | 0.0939 | 2.9209 | importance varies substantially across countries |
| xgboost | UNEMPLOYMENT | traditional_macro | 0.0543 | 1.3055 | importance varies substantially across countries |
| lightgbm | FDI_NET_INFLOWS | capital_flows | 0.0238 | 10.6576 | importance varies substantially across countries |
| lightgbm | UNEMPLOYMENT | traditional_macro | 0.0237 | 2.2361 | importance varies substantially across countries |
| xgboost | MILITARY_EXPENDITURE | military_expenditure | 0.0098 | 2.2361 | importance varies substantially across countries |
| xgboost | HIGH_TECH_EXPORTS | technology_trade | 0.0040 | 2.9161 | importance varies substantially across countries |
| random_forest | MILITARY_EXPENDITURE | military_expenditure | 0.0017 | 1.4199 | importance varies substantially across countries |
| random_forest | UNEMPLOYMENT | traditional_macro | 0.0012 | 11.0205 | importance varies substantially across countries |

## Correlation Notes

| feature_base | feature_group | correlation_with_target | n_rows |
| --- | --- | --- | --- |
| GDP_GROWTH | target_history | 0.5440 | 97 |
| INFLATION | traditional_macro | 0.3739 | 97 |
| FDI_NET_INFLOWS | capital_flows | 0.3717 | 97 |
| UNEMPLOYMENT | traditional_macro | 0.2339 | 97 |
| HIGH_TECH_EXPORTS | technology_trade | 0.0843 | 97 |
| IMPORTS_PERCENT_GDP | trade | -0.0715 | 97 |
| MILITARY_EXPENDITURE | military_expenditure | 0.0693 | 97 |
| EXPORTS_PERCENT_GDP | trade | -0.0549 | 97 |

## Methodological Caveats

- Feature importance is predictive, not causal.
- Correlation is not causation and should not be interpreted as policy evidence by itself.
- The dataset is annual and small, especially after one-year lagging and complete-case filtering.
- Permutation importance is measured on small chronological holdout sets and can be noisy.
- SHAP-style outputs use LightGBM/XGBoost native tree contribution methods where available; RandomForest SHAP was skipped because the `shap` package is not installed.
- Taiwan has no complete World Bank panel rows in the current processed dataset.

## Failed Or Skipped Interpretability Runs

| scope | country | model | reason |
| --- | --- | --- | --- |
| country | TWN | ALL | No complete lagged rows. |
