"""Heatmap for archetype-level statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_heatmap(archetype_ranking: pd.DataFrame, output_dir: Path) -> Path:
    """Draw a lightweight heatmap using matplotlib only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "heatmap_archetypes.png"

    columns = [
        "archetype_reliability_score",
        "best_deck_score",
        "mean_top3_score",
        "support_weight",
        "num_decks",
        "bayesian_mean",
        "herfindahl_index",
        "normalized_entropy",
    ]
    frame = archetype_ranking[["archetype"] + columns].copy().head(20)
    matrix = frame[columns].to_numpy(dtype=float)

    plt.figure(figsize=(12, max(5, 0.42 * len(frame))))
    plt.imshow(matrix, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Normalized value")
    plt.yticks(range(len(frame)), frame["archetype"].tolist())
    plt.xticks(range(len(columns)), columns, rotation=35, ha="right")
    plt.title("Heatmap de estatísticas por arquétipo")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
