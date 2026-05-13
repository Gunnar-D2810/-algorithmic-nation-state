"""Optional LSTM benchmark for GDP growth forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.model_registry import ForecastModelSkipped


def forecast_lstm(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    *,
    random_state: int = 42,
    min_train_size: int = 24,
    epochs: int = 60,
) -> pd.Series:
    """Fit a small LSTM on lagged predictors when TensorFlow is available.

    Scaling is intentionally fit only on the training fold to avoid leakage.
    The model is exploratory because annual macro panels provide few examples.
    """

    if len(train_features) < min_train_size:
        raise ForecastModelSkipped(
            f"LSTM requires at least {min_train_size} training rows; got {len(train_features)}."
        )

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ForecastModelSkipped("TensorFlow is not installed.") from exc

    tf.keras.utils.set_random_seed(random_state)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_x = feature_scaler.fit_transform(train_features.astype(float))
    test_x = feature_scaler.transform(test_features.astype(float))
    train_y = target_scaler.fit_transform(
        train_target.astype(float).to_numpy().reshape(-1, 1)
    )

    train_x_lstm = np.expand_dims(train_x, axis=1)
    test_x_lstm = np.expand_dims(test_x, axis=1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(train_x_lstm.shape[1], train_x_lstm.shape[2])),
            tf.keras.layers.LSTM(12),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_x_lstm, train_y, epochs=epochs, verbose=0)
    scaled_predictions = model.predict(test_x_lstm, verbose=0)
    predictions = target_scaler.inverse_transform(scaled_predictions).reshape(-1)
    return pd.Series(predictions, dtype=float)
