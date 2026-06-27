"""Scatter plot of deck volume, posterior strength, and reliability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(deck_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot games vs observed winrate with reliability encoding."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scatter_reliability.png"

    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(
        deck_frame["jogos"],
        deck_frame["winrate_observed"],
        c=deck_frame["reliability_score"],
        s=30 + deck_frame["posterior_mean"] * 240,
        cmap="viridis",
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Reliability Score")
    plt.xlabel("Jogos")
    plt.ylabel("Winrate observado")
    plt.title("Jogos x Winrate observado")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
