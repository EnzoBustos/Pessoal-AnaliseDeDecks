"""Plot styling helpers for vibrant, consistent figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import AXIS_COLOR, GRID_COLOR, PLOT_BACKGROUND, TITLE_COLOR


def apply_vibrant_theme() -> None:
    """Apply a clean, vibrant matplotlib theme across all figures."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": PLOT_BACKGROUND,
            "axes.facecolor": PLOT_BACKGROUND,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": AXIS_COLOR,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "text.color": TITLE_COLOR,
            "axes.titleweight": "bold",
            "axes.titlecolor": TITLE_COLOR,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.45,
            "legend.frameon": False,
        }
    )


def vibrant_colors(values: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Map numeric values to a vibrant color palette."""

    if values.size == 0:
        return values

    normalized = values.astype(float)
    minimum = float(np.min(normalized))
    maximum = float(np.max(normalized))
    if maximum == minimum:
        normalized = np.zeros_like(normalized)
    else:
        normalized = (normalized - minimum) / (maximum - minimum)
    return plt.colormaps[cmap_name](normalized)
