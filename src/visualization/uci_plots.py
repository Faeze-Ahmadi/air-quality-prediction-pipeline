from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    out_path: Path,
    mae: float | None = None,
    title: str = "Actual vs Predicted",
) -> None:
    """
    Scatter plot of Actual vs Predicted values with y=x reference line.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")

    # Reference line y = x
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal (y = x)")

    plt.xlabel("Actual CO(GT)")
    plt.ylabel("Predicted CO(GT)")

    if mae is not None:
        plt.title(f"{title}\nMAE = {mae:.4f}")
    else:
        plt.title(title)

    plt.legend()
    plt.grid(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_histogram(
    y_true,
    y_pred,
    out_path: Path,
    bins: int = 40,
) -> None:
    """
    Histogram of prediction errors (y_pred - y_true).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    errors = y_pred - y_true

    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=bins, edgecolor="black", alpha=0.7)

    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.grid(axis="y", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
