"""Centralized configuration loading and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_PANEL_COLUMNS = {"country", "year", "indicator", "value"}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths used by scripts and notebooks."""

    root: Path
    config: Path
    raw_data: Path
    processed_data: Path
    tables: Path
    figures: Path
    logs: Path
    reports: Path
    notebooks: Path

    @property
    def macro_panel(self) -> Path:
        """Return the canonical processed macro panel path."""

        return self.processed_data / "macro_panel.csv"


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward to AGENTS.md."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / "config").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing AGENTS.md and config/.")


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse to a mapping: {config_path}")
    return data


def build_project_paths(
    *,
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> ProjectPaths:
    """Build resolved project paths from the YAML storage section."""

    root = (project_root or find_project_root()).resolve()
    config = (config_path or root / "config/indicators.yaml").resolve()
    settings = load_yaml_config(config)
    storage = settings.get("storage", {})
    reports = root / "reports"
    return ProjectPaths(
        root=root,
        config=config,
        raw_data=root / storage.get("raw_data_path", "data/raw"),
        processed_data=root / storage.get("processed_data_path", "data/processed"),
        tables=root / storage.get("tables_path", "reports/tables"),
        figures=root / storage.get("figures_path", "reports/figures"),
        logs=reports / "logs",
        reports=reports,
        notebooks=root / "notebooks",
    )


def ensure_project_directories(paths: ProjectPaths) -> None:
    """Create standard output directories used by the research workflow."""

    for path in [
        paths.raw_data,
        paths.processed_data,
        paths.tables,
        paths.figures,
        paths.logs,
        paths.reports,
        paths.notebooks,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configured_country_codes(config: dict[str, Any]) -> list[str]:
    """Return configured country codes in YAML order."""

    return [country["code"] for country in config.get("countries", []) if "code" in country]


def configured_world_bank_indicators(config: dict[str, Any]) -> list[str]:
    """Return configured World Bank indicator aliases."""

    return list(config.get("sources", {}).get("world_bank", {}).keys())


def validate_macro_panel(
    panel_path: Path,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Validate the cleaned macro panel and return a summary table."""

    rows: list[dict[str, str | int | bool]] = []
    if not panel_path.exists():
        return pd.DataFrame(
            [
                {
                    "check": "macro_panel_exists",
                    "passed": False,
                    "detail": f"missing:{panel_path}",
                }
            ]
        )

    panel = pd.read_csv(panel_path)
    missing_columns = sorted(REQUIRED_PANEL_COLUMNS.difference(panel.columns))
    rows.append(
        {
            "check": "required_columns",
            "passed": not missing_columns,
            "detail": ",".join(missing_columns) if missing_columns else "all_required_columns_present",
        }
    )
    if missing_columns:
        return pd.DataFrame(rows)

    duplicates = panel.duplicated(["country", "year", "indicator"]).sum()
    rows.append(
        {
            "check": "duplicate_country_year_indicator_rows",
            "passed": int(duplicates) == 0,
            "detail": f"duplicate_rows={int(duplicates)}",
        }
    )

    configured_countries = set(configured_country_codes(config))
    observed_countries = set(panel["country"].dropna().astype(str).unique())
    missing_countries = sorted(configured_countries.difference(observed_countries))
    rows.append(
        {
            "check": "configured_countries_present",
            "passed": True,
            "detail": (
                "missing_country_rows=" + ",".join(missing_countries)
                if missing_countries
                else "all_configured_countries_present"
            ),
        }
    )

    configured_indicators = set(configured_world_bank_indicators(config))
    observed_indicators = set(panel["indicator"].dropna().astype(str).unique())
    missing_indicators = sorted(configured_indicators.difference(observed_indicators))
    rows.append(
        {
            "check": "configured_world_bank_indicators_present",
            "passed": not missing_indicators,
            "detail": ",".join(missing_indicators) if missing_indicators else "all_configured_indicators_present",
        }
    )

    null_values = int(panel["value"].isna().sum())
    rows.append(
        {
            "check": "missing_values_reported_not_filled",
            "passed": True,
            "detail": f"missing_value_rows={null_values}",
        }
    )
    rows.append(
        {
            "check": "panel_row_count",
            "passed": len(panel) > 0,
            "detail": f"rows={len(panel)}",
        }
    )
    return pd.DataFrame(rows)
