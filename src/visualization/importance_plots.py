"""Visualization helpers for forecasting interpretability outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_global_feature_importance(global_importance: pd.DataFrame, output_path: Path) -> None:
    """Plot global mean permutation importance by feature."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if global_importance.empty:
        return

    grouped = (
        global_importance.groupby(["feature_base", "feature_group"], as_index=False)
        .agg(importance=("mean_permutation_importance_mean", "mean"))
        .sort_values("importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(grouped["feature_base"], grouped["importance"])
    ax.set_title("Global Lagged Feature Importance")
    ax.set_xlabel("Mean permutation importance (RMSE reduction)")
    ax.set_ylabel("Lagged feature")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_country_feature_importance(by_country: pd.DataFrame, output_dir: Path) -> None:
    """Plot country-level mean permutation importance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if by_country.empty:
        return

    for country, country_frame in by_country.groupby("country"):
        grouped = (
            country_frame.groupby("feature_base", as_index=False)
            .agg(importance=("permutation_importance_mean", "mean"))
            .sort_values("importance", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(grouped["feature_base"], grouped["importance"])
        ax.set_title(f"{country} Lagged Feature Importance")
        ax.set_xlabel("Mean permutation importance (RMSE reduction)")
        ax.set_ylabel("Lagged feature")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{country.lower()}_feature_importance.png", dpi=150)
        plt.close(fig)


def plot_shap_summary(shap_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot native SHAP-style contribution summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if shap_summary.empty:
        return

    for (country, model), frame in shap_summary.groupby(["country", "model"]):
        grouped = (
            frame.groupby("feature_base", as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "mean"))
            .sort_values("mean_abs_shap", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(grouped["feature_base"], grouped["mean_abs_shap"])
        ax.set_title(f"{country} {model} Native Contribution Summary")
        ax.set_xlabel("Mean absolute contribution")
        ax.set_ylabel("Lagged feature")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{country.lower()}_{model}_shap_summary.png", dpi=150)
        plt.close(fig)


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path) -> None:
    """Plot a compact correlation heatmap."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if correlation_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)
    ax.set_title("Lagged Macro Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
