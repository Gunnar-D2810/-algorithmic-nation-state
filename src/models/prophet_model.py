"""Optional Prophet benchmark for annual GDP growth forecasting."""

from __future__ import annotations

import pandas as pd

from src.models.model_registry import ForecastModelSkipped


def forecast_prophet(
    train: pd.DataFrame,
    test_years: pd.Series,
    *,
    min_train_size: int = 10,
) -> pd.Series:
    """Fit Prophet on annual GDP growth and forecast requested years.

    Prophet is optional in this project. If the dependency is absent, callers
    should record the model as skipped rather than fabricating a result.
    """

    if len(train) < min_train_size:
        raise ForecastModelSkipped(
            f"Prophet requires at least {min_train_size} training rows; got {len(train)}."
        )

    try:
        from prophet import Prophet
    except ModuleNotFoundError as exc:
        raise ForecastModelSkipped("Prophet is not installed.") from exc

    prophet_train = pd.DataFrame(
        {
            "ds": pd.to_datetime(train["year"].astype(int).astype(str) + "-01-01"),
            "y": train["GDP_GROWTH"].astype(float),
        }
    )
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_train)

    future = pd.DataFrame(
        {"ds": pd.to_datetime(test_years.astype(int).astype(str) + "-01-01")}
    )
    forecast = model.predict(future)
    return pd.Series(forecast["yhat"].to_numpy(), dtype=float)
