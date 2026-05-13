"""ARIMA baseline model for GDP growth forecasting."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def forecast_arima(
    train: pd.Series,
    *,
    steps: int,
    order: tuple[int, int, int] = (1, 0, 0),
) -> pd.Series:
    """Fit a simple ARIMA model and forecast a fixed number of steps."""

    model = ARIMA(
        train.astype(float),
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return pd.Series(forecast.to_numpy(), dtype=float)
