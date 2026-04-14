"""Wrappers for consistent MLflow logging across all scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import mlflow

TRACKING_URI = f"file:{Path(__file__).resolve().parent.parent / 'mlruns'}"
EXPERIMENT_NAME = "sae-gelu2l-residual"


def get_git_sha() -> str:
    """Return short git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def init_mlflow() -> None:
    """Set tracking URI and experiment. Call once at script start."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(
    run_name: str,
    hook_point: str,
    run_purpose: str,
    tags: dict[str, str] | None = None,
) -> Any:
    """Start an MLflow run with standard tags."""
    all_tags = {
        "git_sha": get_git_sha(),
        "hook_point": hook_point,
        "run_purpose": run_purpose,
    }
    if tags:
        all_tags.update(tags)
    return mlflow.start_run(run_name=run_name, tags=all_tags)


def log_params_dict(params: dict[str, Any]) -> None:
    """Log a flat dict of parameters to the active MLflow run."""
    mlflow.log_params(params)


def log_metrics_dict(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics to the active MLflow run."""
    mlflow.log_metrics(metrics, step=step)
