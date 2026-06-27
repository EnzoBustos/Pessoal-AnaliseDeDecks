"""Scatter plot of deck volume, posterior strength, and reliability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DPI, PLOT_SCATTER_COLOR, SCATTER_COLORBAR_LABEL, SCATTER_CMAP, SCATTER_FIGSIZE


def plot_scatter(deck_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot games vs observed winrate with reliability encoding."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scatter_reliability.png"

    plt.figure(figsize=SCATTER_FIGSIZE)
    scatter = plt.scatter(
        deck_frame["jogos"],
        deck_frame["winrate_observed"],
        c=deck_frame["reliability_score"],
        s=30 + deck_frame["posterior_mean"] * 240,
        cmap=SCATTER_CMAP,
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(scatter, label=SCATTER_COLORBAR_LABEL)
    plt.xlabel("Jogos")
    plt.ylabel("Winrate observado")
    plt.title("Jogos x Winrate observado")
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI)
    plt.close()
    return path
