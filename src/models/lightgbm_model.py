"""LightGBM exploratory benchmark for GDP growth forecasting."""

from __future__ import annotations

import pandas as pd

from src.models.model_registry import ForecastModelSkipped


def forecast_lightgbm(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    *,
    random_state: int = 42,
    min_train_size: int = 8,
) -> pd.Series:
    """Fit a compact LightGBM regressor on lagged macro predictors."""

    if len(train_features) < min_train_size:
        raise ForecastModelSkipped(
            f"LightGBM requires at least {min_train_size} training rows; got {len(train_features)}."
        )

    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ForecastModelSkipped("LightGBM is not installed.") from exc

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=3,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(train_features.astype(float), train_target.astype(float))
    predictions = model.predict(test_features.astype(float))
    return pd.Series(predictions, dtype=float)
