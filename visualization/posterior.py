"""Posterior distribution plots and shrinkage comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import FIG_DPI, POSTERIOR_FIGSIZE, SHRINKAGE_FIGSIZE, PLOT_SECONDARY_COLOR


def plot_posterior_curves(deck_frame: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    """Plot Beta posterior curves for the most reliable decks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "posterior_curves.png"
    frame = deck_frame.sort_values("reliability_score", ascending=False).head(top_n)

    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=POSTERIOR_FIGSIZE)
    for _, row in frame.iterrows():
        pdf = stats.beta.pdf(x, row["posterior_alpha"], row["posterior_beta"])
        label = f"{row['archetype']} #{int(row['deck_rank_in_archetype'])}"
        plt.plot(x, pdf, label=label)

    plt.xlabel("Winrate")
    plt.ylabel("Density")
    plt.title("Distribuições posteriores Beta dos melhores decks")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI)
    plt.close()
    return path


def plot_shrinkage(deck_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Compare observed winrate with posterior mean."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shrinkage.png"

    sorted_frame = deck_frame.sort_values("jogos", ascending=False)
    x = np.arange(len(sorted_frame))

    plt.figure(figsize=SHRINKAGE_FIGSIZE)
    plt.plot(x, sorted_frame["winrate_observed"].to_numpy(dtype=float), label="Winrate observado", linewidth=1.5, color=PLOT_SECONDARY_COLOR)
    plt.plot(x, sorted_frame["posterior_mean"].to_numpy(dtype=float), label="Posterior mean", linewidth=1.5)
    plt.xlabel("Decks ordenados por volume")
    plt.ylabel("Winrate")
    plt.title("Shrinkage: observado vs posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI)
    plt.close()
    return path
