"""Export helpers for markdown reports and publication-style assets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty dataframe."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 10) -> str:
    """Render a small dataframe as a GitHub-flavored markdown table."""

    if frame.empty:
        return "_No rows available._"
    shown = frame.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        rows.append(
            "| "
            + " | ".join(_markdown_cell(row[column]) for column in shown.columns)
            + " |"
        )
    return "\n".join(rows)


def write_markdown(path: Path, content: str) -> None:
    """Write markdown text with parent-directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def table_summary(path: Path) -> dict[str, str | int | bool]:
    """Return lightweight row/column metadata for a CSV file."""

    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "columns": 0}
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return {"path": str(path), "exists": True, "rows": 0, "columns": 0}
    return {
        "path": str(path),
        "exists": True,
        "rows": len(frame),
        "columns": len(frame.columns),
    }


def top_mean_by_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    top_n: int = 8,
    ascending: bool = True,
) -> pd.DataFrame:
    """Return grouped means for compact report tables."""

    if frame.empty or group_column not in frame or value_column not in frame:
        return pd.DataFrame()
    summary = (
        frame.dropna(subset=[value_column])
        .groupby(group_column, as_index=False)
        .agg(mean_value=(value_column, "mean"), n_rows=(value_column, "size"))
        .sort_values("mean_value", ascending=ascending)
        .head(top_n)
    )
    return summary


def _markdown_cell(value) -> str:
    """Format one markdown table cell without relying on optional tabulate."""

    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")
