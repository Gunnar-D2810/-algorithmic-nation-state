"""Fetch configured World Bank macroeconomic indicators."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import fetch_world_bank_macro_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fetch World Bank data from config/indicators.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/indicators.yaml"),
        help="Path to indicator YAML configuration.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root used to resolve configured storage paths.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging for the ingestion script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    """Run the World Bank ingestion pipeline."""

    configure_logging()
    args = parse_args()
    panel, missing_report, missing_by_country = fetch_world_bank_macro_data(
        config_path=args.config,
        project_root=args.project_root,
    )

    logging.info("Fetched %s cleaned World Bank rows.", len(panel))
    logging.info("Missing-value report rows: %s", len(missing_report))
    if missing_by_country:
        logging.warning("Missing configured indicators by country: %s", missing_by_country)


if __name__ == "__main__":
    main()
