"""Cleaning and validation helpers for macroeconomic panel data."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_PANEL_COLUMNS = ["country", "year", "indicator", "value"]


def records_from_world_bank_payload(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one stored World Bank response into canonical long records."""

    response = raw_payload["api_response"]
    observations = response[1] if len(response) > 1 and response[1] is not None else []

    records: list[dict[str, Any]] = []
    for observation in observations:
        year = observation.get("date")
        value = observation.get("value")

        records.append(
            {
                "country": raw_payload["country_code"],
                "year": int(year) if year is not None and str(year).isdigit() else year,
                "indicator": raw_payload["indicator_key"],
                "value": value,
                "country_name": raw_payload["country_name"],
                "indicator_code": raw_payload["indicator_code"],
                "description": raw_payload["description"],
                "unit": raw_payload["unit"],
                "frequency": raw_payload["frequency"],
                "transformation": raw_payload["transformation"],
                "source": "world_bank",
                "retrieved_at": raw_payload["retrieved_at"],
            }
        )

    return records


def build_macro_panel(raw_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a cleaned country-year-indicator panel from raw API payloads."""

    records: list[dict[str, Any]] = []
    for payload in raw_payloads:
        records.extend(records_from_world_bank_payload(payload))

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        return pd.DataFrame(columns=REQUIRED_PANEL_COLUMNS)

    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.sort_values(["country", "indicator", "year"]).reset_index(drop=True)
    return panel


def validate_macro_panel(panel: pd.DataFrame) -> None:
    """Validate required panel columns and duplicate keys.

    Missing values are allowed because they represent real source gaps. They are
    reported separately rather than filled.
    """

    missing_columns = [col for col in REQUIRED_PANEL_COLUMNS if col not in panel.columns]
    if missing_columns:
        raise ValueError(f"Macro panel is missing required columns: {missing_columns}")

    duplicate_mask = panel.duplicated(subset=["country", "year", "indicator"], keep=False)
    if duplicate_mask.any():
        duplicates = panel.loc[duplicate_mask, ["country", "year", "indicator"]]
        raise ValueError(
            "Macro panel contains duplicate country/year/indicator rows: "
            f"{duplicates.head(20).to_dict(orient='records')}"
        )


def missing_value_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by country and indicator."""

    if panel.empty:
        return pd.DataFrame(
            columns=["country", "indicator", "rows", "missing_values", "missing_share"]
        )

    report = (
        panel.groupby(["country", "indicator"], as_index=False)
        .agg(
            rows=("value", "size"),
            missing_values=("value", lambda values: int(values.isna().sum())),
        )
        .assign(missing_share=lambda df: df["missing_values"] / df["rows"])
    )
    return report.sort_values(["country", "indicator"]).reset_index(drop=True)


def missing_indicators_by_country(
    panel: pd.DataFrame,
    countries: list[dict[str, str]],
    indicators: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Find configured indicators with no non-missing values by country."""

    missing: dict[str, list[str]] = {}
    for country in countries:
        country_code = country["code"]
        missing[country_code] = []
        for indicator_key in indicators:
            subset = panel[
                (panel["country"] == country_code) & (panel["indicator"] == indicator_key)
            ]
            if subset.empty or subset["value"].notna().sum() == 0:
                missing[country_code].append(indicator_key)

    return {country: values for country, values in missing.items() if values}
