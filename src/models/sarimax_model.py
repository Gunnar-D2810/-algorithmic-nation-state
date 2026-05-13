"""SARIMAX baseline model for GDP growth forecasting."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_sarimax(
    train: pd.Series,
    train_exog: pd.DataFrame,
    test_exog: pd.DataFrame,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
) -> pd.Series:
    """Fit a simple SARIMAX model using lagged exogenous predictors."""

    model = SARIMAX(
        train.astype(float),
        exog=train_exog.astype(float),
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test_exog), exog=test_exog.astype(float))
    return pd.Series(forecast.to_numpy(), dtype=float)
