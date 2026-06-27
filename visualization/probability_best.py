"""Probability of best deck plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DPI, PROBABILITY_FIGSIZE, VIBRANT_CMAP
from utils.plotting import apply_vibrant_theme, vibrant_colors


def plot_probability_best(deck_frame: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    """Plot the probability of each top deck being the best in its archetype."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "probability_best.png"
    frame = deck_frame.sort_values("prob_best", ascending=False).head(top_n)
    apply_vibrant_theme()

    plt.figure(figsize=PROBABILITY_FIGSIZE)
    labels = [f"{row.archetype} #{int(row.deck_rank_in_archetype)}" for row in frame.itertuples()]
    colors = vibrant_colors(frame["prob_best"].to_numpy(dtype=float), cmap_name=VIBRANT_CMAP)
    plt.bar(labels, frame["prob_best"], color=colors, edgecolor="#0f172a", linewidth=0.8)
    plt.ylabel("Probability of Best")
    plt.xticks(rotation=35, ha="right")
    plt.title("Probabilidade de cada deck ser o melhor do arquétipo")
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI)
    plt.close()
    return path
