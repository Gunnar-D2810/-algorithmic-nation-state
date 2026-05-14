"""Reproducibility and validation helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """Set standard Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)


def collect_environment_metadata(project_root: Path) -> dict[str, str]:
    """Collect lightweight environment metadata for reproducible runs."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(project_root.resolve()),
        "git_commit": _git_value(project_root, ["git", "rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["git", "branch", "--show-current"]),
    }


def write_metadata(path: Path, metadata: dict) -> None:
    """Write run metadata as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_notebook(path: Path) -> dict[str, str | bool | int]:
    """Validate basic notebook structure and ordered code cells."""

    if not path.exists():
        return {"notebook": str(path), "passed": False, "cell_count": 0, "detail": "missing"}
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "notebook": str(path),
            "passed": False,
            "cell_count": 0,
            "detail": f"invalid_json:{exc}",
        }
    cells = notebook.get("cells", [])
    has_kernel = "kernelspec" in notebook.get("metadata", {})
    valid_cells = all("cell_type" in cell and "source" in cell for cell in cells)
    code_cell_positions = [
        index for index, cell in enumerate(cells, start=1) if cell.get("cell_type") == "code"
    ]
    ordered = code_cell_positions == sorted(code_cell_positions)
    return {
        "notebook": str(path),
        "passed": bool(cells and has_kernel and valid_cells and ordered),
        "cell_count": len(cells),
        "detail": "valid_notebook_structure" if cells and has_kernel and valid_cells and ordered else "structure_check_failed",
    }


def validate_imports(module_names: list[str], *, timeout_seconds: int = 120) -> pd.DataFrame:
    """Try importing project modules in bounded subprocesses."""

    rows: list[dict[str, str | bool]] = []
    for module_name in module_names:
        try:
            subprocess.run(
                [sys.executable, "-c", f"import {module_name}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            rows.append({"module": module_name, "passed": True, "detail": "import_ok"})
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    "module": module_name,
                    "passed": False,
                    "detail": f"import_timeout_after_{timeout_seconds}s",
                }
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should catch any import breakage.
            rows.append(
                {
                    "module": module_name,
                    "passed": False,
                    "detail": f"{type(exc).__name__}:{exc}",
                }
            )
    return pd.DataFrame(rows)


def validate_table_schema(
    path: Path,
    required_columns: set[str],
) -> dict[str, str | bool]:
    """Validate that a CSV table exists and includes required columns."""

    if not path.exists():
        return {"table": str(path), "passed": False, "detail": "missing"}
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required_columns.difference(columns))
    return {
        "table": str(path),
        "passed": not missing,
        "detail": ",".join(missing) if missing else "schema_ok",
    }


def _git_value(project_root: Path, command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"
