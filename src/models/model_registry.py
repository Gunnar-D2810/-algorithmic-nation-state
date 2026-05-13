"""Unified registry for baseline and exploratory forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


class ForecastModelSkipped(RuntimeError):
    """Raised when a model should be reported as skipped, not failed."""


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a selectable forecasting model."""

    name: str
    family: str
    requires_exogenous: bool
    min_train_size: int
    dependency_modules: tuple[str, ...] = ()
    exploratory: bool = False
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "naive": ModelSpec(
        name="naive",
        family="baseline",
        requires_exogenous=False,
        min_train_size=8,
        notes="Last-observed-value benchmark used to evaluate added complexity.",
    ),
    "arima": ModelSpec(
        name="arima",
        family="classical",
        requires_exogenous=False,
        min_train_size=8,
        notes="Existing baseline ARIMA model.",
    ),
    "sarimax": ModelSpec(
        name="sarimax",
        family="classical",
        requires_exogenous=True,
        min_train_size=8,
        notes="Existing baseline SARIMAX model with lagged predictors.",
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        family="machine_learning",
        requires_exogenous=True,
        min_train_size=8,
        notes="Existing baseline RandomForest model with lagged predictors.",
    ),
    "prophet": ModelSpec(
        name="prophet",
        family="modern_time_series",
        requires_exogenous=False,
        min_train_size=10,
        dependency_modules=("prophet",),
        exploratory=True,
        notes="Exploratory annual univariate Prophet benchmark.",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        family="gradient_boosting",
        requires_exogenous=True,
        min_train_size=8,
        dependency_modules=("lightgbm",),
        exploratory=True,
        notes="Exploratory gradient-boosted tree benchmark using lagged predictors.",
    ),
    "lstm": ModelSpec(
        name="lstm",
        family="deep_learning",
        requires_exogenous=True,
        min_train_size=24,
        dependency_modules=("tensorflow",),
        exploratory=True,
        notes="Exploratory LSTM benchmark; annual country samples are likely sparse.",
    ),
    "optional_nbeats": ModelSpec(
        name="optional_nbeats",
        family="deep_learning",
        requires_exogenous=False,
        min_train_size=36,
        dependency_modules=("neuralforecast", "torch"),
        exploratory=True,
        notes="Skipped unless NeuralForecast/Torch are installed and sample size is adequate.",
    ),
    "optional_nhits": ModelSpec(
        name="optional_nhits",
        family="deep_learning",
        requires_exogenous=False,
        min_train_size=36,
        dependency_modules=("neuralforecast", "torch"),
        exploratory=True,
        notes="Skipped unless NeuralForecast/Torch are installed and sample size is adequate.",
    ),
    "optional_tft_or_patchtst": ModelSpec(
        name="optional_tft_or_patchtst",
        family="deep_learning",
        requires_exogenous=True,
        min_train_size=80,
        dependency_modules=("neuralforecast", "torch"),
        exploratory=True,
        notes="Transformer-style models require more annual observations than this panel provides.",
    ),
}

BASELINE_MODEL_NAMES = ("arima", "sarimax", "random_forest")
MODERN_MODEL_NAMES = (
    "prophet",
    "lightgbm",
    "lstm",
    "optional_nbeats",
    "optional_nhits",
    "optional_tft_or_patchtst",
)


def get_model_registry() -> dict[str, ModelSpec]:
    """Return a copy of the model registry."""

    return dict(MODEL_REGISTRY)


def dependency_status(model_name: str) -> tuple[bool, str]:
    """Check whether optional dependencies for a registered model are importable."""

    spec = MODEL_REGISTRY[model_name]
    missing = [module for module in spec.dependency_modules if find_spec(module) is None]
    if missing:
        return False, f"Missing optional dependency modules: {', '.join(missing)}"
    return True, "available"


def parse_model_selection(selection: str | None) -> list[str]:
    """Parse a comma-separated model selection string."""

    if selection is None or selection.strip().lower() == "all":
        return list(MODERN_MODEL_NAMES)

    selected = [name.strip().lower() for name in selection.split(",") if name.strip()]
    unknown = [name for name in selected if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return selected
