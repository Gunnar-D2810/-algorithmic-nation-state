"""Feature-importance methods for tree-based GDP forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class FittedTreeModel:
    """Container for a fitted tree model and its metadata."""

    name: str
    model: Any
    feature_columns: list[str]
    train_rows: int
    test_rows: int


def fit_random_forest(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    random_state: int = 42,
) -> FittedTreeModel:
    """Fit the RandomForest model used for interpretation."""

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features.astype(float), train_target.astype(float))
    return FittedTreeModel(
        name="random_forest",
        model=model,
        feature_columns=list(train_features.columns),
        train_rows=len(train_features),
        test_rows=0,
    )


def fit_lightgbm(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    random_state: int = 42,
) -> FittedTreeModel:
    """Fit a compact LightGBM model if the dependency is available."""

    if find_spec("lightgbm") is None:
        raise ModuleNotFoundError("lightgbm")

    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=3,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(train_features.astype(float), train_target.astype(float))
    return FittedTreeModel(
        name="lightgbm",
        model=model,
        feature_columns=list(train_features.columns),
        train_rows=len(train_features),
        test_rows=0,
    )


def fit_xgboost(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    random_state: int = 42,
) -> FittedTreeModel:
    """Fit a compact XGBoost model if the dependency is available."""

    if find_spec("xgboost") is None:
        raise ModuleNotFoundError("xgboost")

    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(train_features.astype(float), train_target.astype(float))
    return FittedTreeModel(
        name="xgboost",
        model=model,
        feature_columns=list(train_features.columns),
        train_rows=len(train_features),
        test_rows=0,
    )


def model_native_importance(fitted: FittedTreeModel) -> pd.DataFrame:
    """Extract model-native feature importance where available."""

    importances = getattr(fitted.model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "native_importance"])

    return pd.DataFrame(
        {
            "feature": fitted.feature_columns,
            "native_importance": np.asarray(importances, dtype=float),
        }
    )


def permutation_importance_table(
    fitted: FittedTreeModel,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    *,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on chronological holdout rows."""

    if test_features.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
            ]
        )

    result = permutation_importance(
        fitted.model,
        test_features.astype(float),
        test_target.astype(float),
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )
    return pd.DataFrame(
        {
            "feature": fitted.feature_columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    )


def shap_contribution_table(
    fitted: FittedTreeModel,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Compute SHAP-style contributions when practical.

    The `shap` package is not required. LightGBM and XGBoost expose native
    contribution outputs that are SHAP-like for tree models; RandomForest is
    skipped unless a dedicated SHAP dependency is added later.
    """

    if fitted.name == "lightgbm":
        contributions = fitted.model.predict(features.astype(float), pred_contrib=True)
        method = "lightgbm_native_pred_contrib"
    elif fitted.name == "xgboost":
        booster = fitted.model.get_booster()
        try:
            from xgboost import DMatrix
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("xgboost") from exc
        contributions = booster.predict(
            DMatrix(features.astype(float), feature_names=fitted.feature_columns),
            pred_contribs=True,
        )
        method = "xgboost_native_pred_contribs"
    else:
        return (
            pd.DataFrame(
                columns=["feature", "mean_abs_shap", "mean_shap", "shap_method"]
            ),
            "skipped_no_native_shap_contributions",
        )

    feature_contributions = np.asarray(contributions)[:, : len(fitted.feature_columns)]
    table = pd.DataFrame(
        {
            "feature": fitted.feature_columns,
            "mean_abs_shap": np.abs(feature_contributions).mean(axis=0),
            "mean_shap": feature_contributions.mean(axis=0),
            "shap_method": method,
        }
    )
    return table, method


def merge_importance_tables(
    *,
    native: pd.DataFrame,
    permutation: pd.DataFrame,
    shap_values: pd.DataFrame,
) -> pd.DataFrame:
    """Merge native, permutation, and SHAP-style importance views."""

    merged = native.merge(permutation, on="feature", how="outer")
    merged = merged.merge(shap_values, on="feature", how="outer")
    for column in [
        "native_importance",
        "permutation_importance_mean",
        "permutation_importance_std",
        "mean_abs_shap",
        "mean_shap",
    ]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    if "shap_method" not in merged.columns:
        merged["shap_method"] = ""
    return merged


def normalize_feature_name(feature: str) -> str:
    """Remove the one-year lag suffix for grouped reporting."""

    return feature.removesuffix("_L1")


def feature_group(feature: str) -> str:
    """Map model feature names to macroeconomic interpretation groups."""

    normalized = normalize_feature_name(feature)
    groups = {
        "GDP_GROWTH": "target_history",
        "INFLATION": "traditional_macro",
        "UNEMPLOYMENT": "traditional_macro",
        "EXPORTS_PERCENT_GDP": "trade",
        "IMPORTS_PERCENT_GDP": "trade",
        "FDI_NET_INFLOWS": "capital_flows",
        "HIGH_TECH_EXPORTS": "technology_trade",
        "MILITARY_EXPENDITURE": "military_expenditure",
    }
    return groups.get(normalized, "other")
