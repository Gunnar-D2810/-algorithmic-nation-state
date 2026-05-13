"""Config-driven macroeconomic data loading orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.data_cleaning import (
    build_macro_panel,
    missing_indicators_by_country,
    missing_value_report,
    validate_macro_panel,
)
from src.data.world_bank_client import WorldBankClient

LOGGER = logging.getLogger(__name__)


def load_indicator_config(config_path: Path) -> dict[str, Any]:
    """Load the indicator YAML configuration."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")
    return config


def configured_countries(config: dict[str, Any]) -> list[dict[str, str]]:
    """Return countries from the project config."""

    countries = config.get("countries", [])
    if not countries:
        raise ValueError("No countries configured under 'countries'.")
    return countries


def configured_world_bank_indicators(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return World Bank indicator mappings from the project config."""

    indicators = config.get("sources", {}).get("world_bank", {})
    if not indicators:
        raise ValueError("No World Bank indicators configured under 'sources.world_bank'.")
    return indicators


def save_raw_world_bank_response(
    raw_dir: Path,
    raw_payload: dict[str, Any],
) -> Path:
    """Persist one raw World Bank response with request metadata."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{raw_payload['country_code']}__"
        f"{raw_payload['indicator_key']}__"
        f"{raw_payload['indicator_code'].replace('.', '_')}.json"
    )
    output_path = raw_dir / filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(raw_payload, file, indent=2, sort_keys=True)
    return output_path


def fetch_world_bank_macro_data(
    *,
    config_path: Path,
    project_root: Path,
    client: WorldBankClient | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Fetch configured World Bank indicators and write raw/processed outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]
        The cleaned panel, missing-value report, and configured indicators that
        have no observed values by country.
    """

    config = load_indicator_config(config_path)
    countries = configured_countries(config)
    indicators = configured_world_bank_indicators(config)

    storage_config = config.get("storage", {})
    raw_base = project_root / storage_config.get("raw_data_path", "data/raw")
    processed_base = project_root / storage_config.get(
        "processed_data_path", "data/processed"
    )
    raw_world_bank_dir = raw_base / "world_bank"
    processed_base.mkdir(parents=True, exist_ok=True)

    wb_client = client or WorldBankClient()
    raw_payloads: list[dict[str, Any]] = []

    for country in countries:
        country_code = country["code"]
        country_name = country.get("name", country_code)

        for indicator_key, indicator_config in indicators.items():
            indicator_code = indicator_config["indicator"]
            retrieved_at = datetime.now(timezone.utc).isoformat()
            response = wb_client.fetch_indicator(country_code, indicator_code)

            raw_payload = {
                "retrieved_at": retrieved_at,
                "country_code": country_code,
                "country_name": country_name,
                "indicator_key": indicator_key,
                "indicator_code": indicator_code,
                "description": indicator_config.get("description", ""),
                "unit": indicator_config.get("unit", ""),
                "frequency": indicator_config.get("frequency", ""),
                # The first ingestion version records configured transformations
                # as metadata but does not transform values implicitly.
                "transformation": indicator_config.get("transformation", "none"),
                "api_response": response,
            }
            output_path = save_raw_world_bank_response(raw_world_bank_dir, raw_payload)
            LOGGER.info("Saved raw World Bank response to %s", output_path)
            raw_payloads.append(raw_payload)

    panel = build_macro_panel(raw_payloads)
    validate_macro_panel(panel)

    panel_path = processed_base / "macro_panel.csv"
    panel.to_csv(panel_path, index=False)
    LOGGER.info("Saved cleaned macro panel to %s (%s rows)", panel_path, len(panel))

    missing_report = missing_value_report(panel)
    missing_report_path = processed_base / "macro_panel_missing_report.csv"
    missing_report.to_csv(missing_report_path, index=False)
    LOGGER.info("Saved missing-value report to %s", missing_report_path)

    missing_by_country = missing_indicators_by_country(panel, countries, indicators)
    missing_by_country_path = processed_base / "missing_indicators_by_country.json"
    with missing_by_country_path.open("w", encoding="utf-8") as file:
        json.dump(missing_by_country, file, indent=2, sort_keys=True)
    LOGGER.info("Saved missing-indicator report to %s", missing_by_country_path)

    if missing_by_country:
        LOGGER.warning("Some country/indicator pairs have no observed values: %s", missing_by_country)

    return panel, missing_report, missing_by_country
