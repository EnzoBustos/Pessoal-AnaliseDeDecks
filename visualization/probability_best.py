"""Probability of best deck plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_probability_best(deck_frame: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    """Plot the probability of each top deck being the best in its archetype."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "probability_best.png"
    frame = deck_frame.sort_values("prob_best", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    labels = [f"{row.archetype} #{int(row.deck_rank_in_archetype)}" for row in frame.itertuples()]
    plt.bar(labels, frame["prob_best"], color="#f59e0b")
    plt.ylabel("Probability of Best")
    plt.xticks(rotation=35, ha="right")
    plt.title("Probabilidade de cada deck ser o melhor do arquétipo")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
