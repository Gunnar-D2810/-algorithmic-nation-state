"""Run the integrated Algorithmic Nation-State research workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import (
    build_project_paths,
    ensure_project_directories,
    load_yaml_config,
    validate_macro_panel,
)
from src.utils.logging_utils import configure_project_logging, log_stage, timestamp_slug
from src.utils.reproducibility import (
    collect_environment_metadata,
    set_global_seed,
    validate_imports,
    validate_notebook,
    validate_table_schema,
    write_metadata,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Status record for one pipeline stage."""

    stage: str
    status: str
    seconds: float
    detail: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the full reproducible analysis pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/indicators.yaml")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-steps", type=int, default=30)
    parser.add_argument("--monte-carlo-iterations", type=int, default=100)
    parser.add_argument("--refresh-data", action="store_true", help="Fetch World Bank data before validation.")
    parser.add_argument("--skip-baseline-forecasting", action="store_true")
    parser.add_argument("--skip-modern-forecasting", action="store_true")
    parser.add_argument("--skip-interpretability", action="store_true")
    parser.add_argument("--skip-abm", action="store_true")
    parser.add_argument("--skip-bayesian", action="store_true")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--skip-report-assets", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end research workflow."""

    args = parse_args()
    paths = build_project_paths(project_root=args.project_root, config_path=args.config)
    ensure_project_directories(paths)
    logger, log_path = configure_project_logging(logs_dir=paths.logs, name="full_pipeline")
    logger.info("Full pipeline log: %s", log_path)
    set_global_seed(args.seed)
    config = load_yaml_config(paths.config)

    metadata = collect_environment_metadata(paths.root)
    metadata.update(
        {
            "seed": args.seed,
            "time_steps": args.time_steps,
            "monte_carlo_iterations": args.monte_carlo_iterations,
            "config_path": str(paths.config),
            "pipeline_log": str(log_path),
        }
    )
    write_metadata(paths.reports / "full_pipeline_metadata.json", metadata)

    results: list[StageResult] = []
    run_id = timestamp_slug()

    def execute(stage: str, command: list[str] | None = None, action=None) -> None:
        start = time.perf_counter()
        try:
            log_stage(logger, stage, "start")
            if args.dry_run:
                detail = "dry_run"
            elif action is not None:
                detail = action()
            elif command is not None:
                completed = subprocess.run(
                    command,
                    cwd=paths.root,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                detail = completed.stdout.strip()[-500:] if completed.stdout.strip() else "completed"
            else:
                detail = "no_op"
            seconds = time.perf_counter() - start
            results.append(StageResult(stage, "passed", seconds, detail))
            log_stage(logger, stage, "passed", f"{seconds:.2f}s")
        except Exception as exc:  # noqa: BLE001 - stage runner should record all failures.
            seconds = time.perf_counter() - start
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                detail = _subprocess_detail(exc)
            results.append(StageResult(stage, "failed", seconds, detail))
            logger.exception("Stage failed: %s", stage)
            if not args.continue_on_error:
                write_pipeline_status(paths, results, run_id)
                raise

    if args.refresh_data:
        execute(
            "world_bank_ingestion",
            [
                str(args.python),
                "scripts/fetch_world_bank_data.py",
                "--config",
                str(paths.config),
                "--project-root",
                str(paths.root),
            ],
        )

    execute("ingestion_validation", action=lambda: validate_ingestion(paths, config))
    execute("import_validation", action=lambda: validate_project_imports(paths))
    execute("schema_validation", action=lambda: validate_core_schemas(paths))

    if not args.skip_baseline_forecasting:
        execute("baseline_forecasting", [str(args.python), "scripts/run_forecasting.py"])
    if not args.skip_modern_forecasting:
        execute("modern_forecasting", [str(args.python), "scripts/run_modern_forecasting.py"])
    if not args.skip_interpretability:
        execute("interpretability", [str(args.python), "scripts/run_interpretability.py"])
    if not args.skip_abm:
        execute(
            "abm",
            [
                str(args.python),
                "scripts/run_abm.py",
                "--seed",
                str(args.seed),
                "--time-steps",
                str(args.time_steps),
            ],
        )
    if not args.skip_bayesian:
        execute("bayesian_updates", [str(args.python), "scripts/run_bayesian_updates.py"])
    if not args.skip_monte_carlo:
        execute(
            "monte_carlo",
            [
                str(args.python),
                "scripts/run_monte_carlo.py",
                "--seed",
                str(args.seed),
                "--time-steps",
                str(args.time_steps),
                "--iterations",
                str(args.monte_carlo_iterations),
            ],
        )
    if not args.skip_report_assets:
        execute(
            "report_assets",
            [
                str(args.python),
                "scripts/generate_report_assets.py",
                "--config",
                str(paths.config),
                "--project-root",
                str(paths.root),
            ],
        )

    execute("notebook_validation", action=lambda: validate_notebooks(paths))
    write_pipeline_status(paths, results, run_id)
    logger.info("Full pipeline complete. Status written to reports/tables.")


def validate_ingestion(paths, config: dict) -> str:
    """Validate the existing processed macro panel."""

    summary = validate_macro_panel(paths.macro_panel, config=config)
    output = paths.tables / "ingestion_validation_summary.csv"
    summary.to_csv(output, index=False)
    failed = summary.loc[~summary["passed"].astype(bool)]
    return f"checks={len(summary)};failed={len(failed)};output={output}"


def validate_project_imports(paths) -> str:
    """Validate imports for core project modules."""

    modules = [
        "src.data.data_loader",
        "src.models.evaluation",
        "src.models.modern_forecasting",
        "src.models.interpretability",
        "src.abm.simulation",
        "src.bayesian.forecast_updates",
        "src.monte_carlo.simulation_engine",
        "src.utils.config_loader",
    ]
    frame = validate_imports(modules)
    output = paths.tables / "import_validation_summary.csv"
    frame.to_csv(output, index=False)
    failed = frame.loc[~frame["passed"].astype(bool)]
    if not failed.empty:
        raise RuntimeError(f"Import validation failed: {failed.to_dict('records')}")
    return f"modules={len(frame)};output={output}"


def validate_core_schemas(paths) -> str:
    """Validate core output schemas."""

    checks = [
        validate_table_schema(paths.macro_panel, {"country", "year", "indicator", "value"}),
        validate_table_schema(paths.tables / "forecast_model_comparison.csv", {"country", "model", "rmse", "mae"}),
        validate_table_schema(paths.tables / "feature_importance_global.csv", {"model", "feature_base"}),
        validate_table_schema(paths.tables / "abm/abm_timeseries.csv", {"scenario", "timestep"}),
        validate_table_schema(paths.tables / "probabilistic/bayesian_posteriors.csv", {"event_name", "posterior_mean"}),
        validate_table_schema(paths.tables / "probabilistic/monte_carlo_summary.csv", {"scenario", "metric", "mean"}),
    ]
    frame = pd.DataFrame(checks)
    output = paths.tables / "schema_validation_summary.csv"
    frame.to_csv(output, index=False)
    failed = frame.loc[~frame["passed"].astype(bool)]
    if not failed.empty:
        raise RuntimeError(f"Schema validation failed: {failed.to_dict('records')}")
    return f"tables={len(frame)};output={output}"


def validate_notebooks(paths) -> str:
    """Validate requested notebooks for basic structure."""

    notebook_paths = [
        paths.notebooks / "main_analysis.ipynb",
        paths.notebooks / "forecasting_analysis.ipynb",
        paths.notebooks / "abm_analysis.ipynb",
        paths.notebooks / "probabilistic_analysis.ipynb",
    ]
    frame = pd.DataFrame([validate_notebook(path) for path in notebook_paths])
    output = paths.tables / "notebook_validation_summary.csv"
    frame.to_csv(output, index=False)
    failed = frame.loc[~frame["passed"].astype(bool)]
    if not failed.empty:
        raise RuntimeError(f"Notebook validation failed: {failed.to_dict('records')}")
    return f"notebooks={len(frame)};output={output}"


def write_pipeline_status(paths, results: list[StageResult], run_id: str) -> None:
    """Write timestamped and latest pipeline status CSVs."""

    frame = pd.DataFrame([result.__dict__ for result in results])
    status_path = paths.logs / f"full_pipeline_status_{run_id}.csv"
    latest_path = paths.tables / "pipeline_status_latest.csv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(status_path, index=False)
    frame.to_csv(latest_path, index=False)


def _subprocess_detail(exc: subprocess.CalledProcessError) -> str:
    stdout = (exc.stdout or "").strip()
    stderr = (exc.stderr or "").strip()
    detail = stderr or stdout or str(exc)
    return detail[-1000:]


if __name__ == "__main__":
    main()
