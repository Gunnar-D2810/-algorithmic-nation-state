"""Shared logging helpers for reproducible scripts."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path


def timestamp_slug() -> str:
    """Return a UTC timestamp suitable for filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def configure_project_logging(
    *,
    logs_dir: Path,
    name: str,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Configure stream and file logging for a project script."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}_{timestamp_slug()}.log"
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logging.getLogger(name), log_path


def log_stage(logger: logging.Logger, stage: str, status: str, detail: str = "") -> None:
    """Log a standardized pipeline stage message."""

    suffix = f" - {detail}" if detail else ""
    logger.info("[%s] %s%s", status.upper(), stage, suffix)
