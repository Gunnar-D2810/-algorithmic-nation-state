"""RandomForest baseline model for GDP growth forecasting."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def forecast_random_forest(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.Series:
    """Fit a transparent RandomForest baseline and predict test rows."""

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features.astype(float), train_target.astype(float))
    predictions = model.predict(test_features.astype(float))
    return pd.Series(predictions, dtype=float)
