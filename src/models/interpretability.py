"""Interpretability workflow for tree-based GDP forecasting models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.models.feature_importance import (
    FittedTreeModel,
    feature_group,
    fit_lightgbm,
    fit_random_forest,
    fit_xgboost,
    merge_importance_tables,
    model_native_importance,
    normalize_feature_name,
    permutation_importance_table,
    shap_contribution_table,
)
from src.models.modern_forecasting import (
    FEATURE_INDICATORS,
    TARGET_INDICATOR,
    add_lagged_features,
    lag_feature_columns,
    load_macro_panel,
    macro_panel_to_wide,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterpretabilityDataset:
    """Prepared feature matrix for interpretability."""

    wide: pd.DataFrame
    feature_columns: list[str]


def prepare_interpretability_dataset(panel_path: Path) -> InterpretabilityDataset:
    """Load macro panel and create one-year lagged features."""

    panel = load_macro_panel(panel_path)
    wide = macro_panel_to_wide(panel, [TARGET_INDICATOR, *FEATURE_INDICATORS])
    lagged_frames = [
        add_lagged_features(country_frame)
        for _, country_frame in wide.groupby("country", sort=True)
    ]
    lagged = pd.concat(lagged_frames, ignore_index=True) if lagged_frames else pd.DataFrame()
    feature_columns = lag_feature_columns()
    validate_no_target_leakage(feature_columns)
    return InterpretabilityDataset(wide=lagged, feature_columns=feature_columns)


def complete_model_frame(
    wide: pd.DataFrame,
    feature_columns: list[str],
    *,
    country: str | None = None,
    train_start_year: int = 2000,
) -> pd.DataFrame:
    """Return complete target/feature rows for one country or pooled panel."""

    frame = wide.copy()
    if country is not None:
        frame = frame.loc[frame["country"] == country].copy()
    required_columns = ["country", "year", TARGET_INDICATOR, *feature_columns]
    if frame.empty:
        return pd.DataFrame(columns=required_columns)
    frame = (
        frame.loc[frame["year"] >= train_start_year, required_columns]
        .dropna(subset=[TARGET_INDICATOR, *feature_columns])
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    return frame


def chronological_train_test_split(
    frame: pd.DataFrame,
    *,
    test_size: int = 3,
    country_level: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so test years are strictly after train years."""

    if frame.empty or len(frame) <= test_size:
        return frame.iloc[0:0].copy(), frame.iloc[0:0].copy()

    if country_level:
        sorted_frame = frame.sort_values("year").reset_index(drop=True)
        train = sorted_frame.iloc[:-test_size].copy()
        test = sorted_frame.iloc[-test_size:].copy()
        if not train.empty and not test.empty and train["year"].max() >= test["year"].min():
            raise ValueError("Country-level train/test split violates chronology.")
        return train, test

    max_test_years = sorted(frame["year"].dropna().unique())[-test_size:]
    train = frame.loc[~frame["year"].isin(max_test_years)].copy()
    test = frame.loc[frame["year"].isin(max_test_years)].copy()
    if not train.empty and not test.empty and train["year"].max() >= test["year"].min():
        raise ValueError("Global train/test split violates chronology.")
    return train, test


def validate_no_target_leakage(feature_columns: list[str]) -> None:
    """Ensure all features are lagged and no contemporaneous target is present."""

    for feature in feature_columns:
        if feature == TARGET_INDICATOR:
            raise ValueError("Contemporaneous target is included as a feature.")
        if not feature.endswith("_L1"):
            raise ValueError(f"Feature is not explicitly lagged: {feature}")


def build_model_factories() -> dict[str, Callable[[pd.DataFrame, pd.Series], FittedTreeModel]]:
    """Return model factories available for interpretability."""

    return {
        "random_forest": lambda x, y: fit_random_forest(x, y),
        "lightgbm": lambda x, y: fit_lightgbm(x, y),
        "xgboost": lambda x, y: fit_xgboost(x, y),
    }


def compute_importance_for_frame(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    scope: str,
    country: str,
    train_start_year: int,
    test_size: int,
    min_train_rows: int,
    country_level: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Fit tree models and compute importance tables for a prepared frame."""

    failures: list[dict[str, str]] = []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), [
            {"scope": scope, "country": country, "model": "ALL", "reason": "No complete lagged rows."}
        ]

    train, test = chronological_train_test_split(
        frame,
        test_size=test_size,
        country_level=country_level,
    )
    if len(train) < min_train_rows or test.empty:
        return pd.DataFrame(), pd.DataFrame(), [
            {
                "scope": scope,
                "country": country,
                "model": "ALL",
                "reason": (
                    f"Insufficient chronological split rows: train={len(train)}, test={len(test)}."
                ),
            }
        ]

    importance_rows: list[pd.DataFrame] = []
    shap_rows: list[pd.DataFrame] = []
    model_factories = build_model_factories()
    for model_name, factory in model_factories.items():
        try:
            fitted = factory(train[feature_columns], train[TARGET_INDICATOR])
            fitted = FittedTreeModel(
                name=fitted.name,
                model=fitted.model,
                feature_columns=fitted.feature_columns,
                train_rows=len(train),
                test_rows=len(test),
            )
            native = model_native_importance(fitted)
            permutation = permutation_importance_table(
                fitted,
                test[feature_columns],
                test[TARGET_INDICATOR],
            )
            shap_values, shap_method = shap_contribution_table(
                fitted,
                test[feature_columns],
            )
            merged = merge_importance_tables(
                native=native,
                permutation=permutation,
                shap_values=shap_values,
            )
            merged["scope"] = scope
            merged["country"] = country
            merged["model"] = model_name
            merged["train_start_year"] = int(train["year"].min())
            merged["train_end_year"] = int(train["year"].max())
            merged["test_start_year"] = int(test["year"].min())
            merged["test_end_year"] = int(test["year"].max())
            merged["n_train"] = len(train)
            merged["n_test"] = len(test)
            merged["feature_base"] = merged["feature"].map(normalize_feature_name)
            merged["feature_group"] = merged["feature"].map(feature_group)
            merged["importance_warning"] = merged.apply(_importance_warning, axis=1)
            importance_rows.append(merged)

            if not shap_values.empty:
                shap_output = shap_values.copy()
                shap_output["scope"] = scope
                shap_output["country"] = country
                shap_output["model"] = model_name
                shap_output["shap_method"] = shap_method
                shap_output["feature_base"] = shap_output["feature"].map(normalize_feature_name)
                shap_output["feature_group"] = shap_output["feature"].map(feature_group)
                shap_rows.append(shap_output)
        except Exception as exc:  # noqa: BLE001 - keep other models/scopes running.
            LOGGER.warning("Importance failed for %s/%s/%s: %s", scope, country, model_name, exc)
            failures.append(
                {
                    "scope": scope,
                    "country": country,
                    "model": model_name,
                    "reason": str(exc),
                }
            )

    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    shap_summary = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    return importance, shap_summary, failures


def compute_correlation_outputs(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute global correlation matrix and feature-target correlations."""

    columns = [TARGET_INDICATOR, *feature_columns]
    matrix = frame[columns].corr(numeric_only=True)
    summary_rows: list[dict[str, float | str]] = []
    for country, country_frame in frame.groupby("country"):
        for feature in feature_columns:
            summary_rows.append(
                {
                    "country": country,
                    "feature": feature,
                    "feature_base": normalize_feature_name(feature),
                    "feature_group": feature_group(feature),
                    "correlation_with_target": country_frame[[TARGET_INDICATOR, feature]]
                    .corr()
                    .iloc[0, 1],
                    "n_rows": int(country_frame[[TARGET_INDICATOR, feature]].dropna().shape[0]),
                }
            )
    for feature in feature_columns:
        summary_rows.append(
            {
                "country": "GLOBAL",
                "feature": feature,
                "feature_base": normalize_feature_name(feature),
                "feature_group": feature_group(feature),
                "correlation_with_target": frame[[TARGET_INDICATOR, feature]].corr().iloc[0, 1],
                "n_rows": int(frame[[TARGET_INDICATOR, feature]].dropna().shape[0]),
            }
        )
    return matrix, pd.DataFrame(summary_rows)


def compute_lag_relationships(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Summarize lagged feature-target relationships by country and globally."""

    rows: list[dict[str, float | str | int]] = []
    for scope_name, scope_frame in [("GLOBAL", frame)] + list(frame.groupby("country")):
        for feature in feature_columns:
            pair = scope_frame[[TARGET_INDICATOR, feature]].dropna()
            corr = pair.corr().iloc[0, 1] if len(pair) >= 3 else np.nan
            rows.append(
                {
                    "scope": scope_name,
                    "feature": feature,
                    "feature_base": normalize_feature_name(feature),
                    "feature_group": feature_group(feature),
                    "lag_years": 1,
                    "correlation_with_target": corr,
                    "n_rows": len(pair),
                }
            )
    return pd.DataFrame(rows)


def summarize_global_importance(by_country: pd.DataFrame) -> pd.DataFrame:
    """Aggregate country/model importance into a global summary."""

    if by_country.empty:
        return pd.DataFrame()

    metric_columns = [
        "native_importance",
        "permutation_importance_mean",
        "permutation_importance_std",
        "mean_abs_shap",
    ]
    summary = (
        by_country.groupby(["model", "feature", "feature_base", "feature_group"], as_index=False)
        .agg(
            **{f"mean_{col}": (col, "mean") for col in metric_columns},
            countries=("country", "nunique"),
        )
    )
    summary["instability_score"] = summary.apply(
        lambda row: _instability_score(by_country, row["model"], row["feature"]),
        axis=1,
    )
    summary["stability_warning"] = np.where(
        summary["instability_score"] > 1.0,
        "importance varies substantially across countries",
        "",
    )
    return summary.sort_values(
        ["mean_permutation_importance_mean", "mean_native_importance"],
        ascending=False,
    ).reset_index(drop=True)


def rolling_importance(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    min_train_rows: int = 8,
    test_size: int = 3,
) -> pd.DataFrame:
    """Compute expanding-window RandomForest permutation importance by country."""

    rows: list[pd.DataFrame] = []
    for country, country_frame in frame.groupby("country"):
        country_frame = country_frame.sort_values("year").reset_index(drop=True)
        if len(country_frame) < min_train_rows + test_size:
            continue
        for test_end in range(min_train_rows + test_size, len(country_frame) + 1, test_size):
            train = country_frame.iloc[: test_end - test_size]
            test = country_frame.iloc[test_end - test_size : test_end]
            fitted = fit_random_forest(train[feature_columns], train[TARGET_INDICATOR])
            importance = permutation_importance_table(
                fitted,
                test[feature_columns],
                test[TARGET_INDICATOR],
                n_repeats=20,
            )
            importance["country"] = country
            importance["model"] = "random_forest"
            importance["train_end_year"] = int(train["year"].max())
            importance["test_start_year"] = int(test["year"].min())
            importance["test_end_year"] = int(test["year"].max())
            importance["feature_base"] = importance["feature"].map(normalize_feature_name)
            importance["feature_group"] = importance["feature"].map(feature_group)
            rows.append(importance)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def data_quality_summary(frame: pd.DataFrame, country_codes: list[str]) -> pd.DataFrame:
    """Summarize complete lagged rows by configured country."""

    rows: list[dict[str, int | str]] = []
    for country in country_codes:
        country_frame = frame.loc[frame["country"] == country]
        rows.append(
            {
                "country": country,
                "complete_lagged_rows": int(len(country_frame)),
                "first_complete_year": int(country_frame["year"].min()) if not country_frame.empty else "",
                "last_complete_year": int(country_frame["year"].max()) if not country_frame.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _importance_warning(row: pd.Series) -> str:
    """Classify weak or unstable feature importance rows."""

    mean_value = row.get("permutation_importance_mean", 0.0)
    std_value = row.get("permutation_importance_std", 0.0)
    if mean_value <= 0:
        return "permutation importance is non-positive on holdout rows"
    if std_value > abs(mean_value):
        return "permutation importance has high repeat variability"
    return ""


def _instability_score(by_country: pd.DataFrame, model: str, feature: str) -> float:
    """Coefficient-of-variation style instability score across countries."""

    values = by_country.loc[
        (by_country["model"] == model) & (by_country["feature"] == feature),
        "permutation_importance_mean",
    ].to_numpy(dtype=float)
    if len(values) <= 1:
        return float("nan")
    denominator = abs(np.nanmean(values))
    if denominator == 0 or np.isnan(denominator):
        return float("inf")
    return float(np.nanstd(values) / denominator)
