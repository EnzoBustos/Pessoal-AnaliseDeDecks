"""Forest plot for posterior credible intervals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forest(deck_frame: pd.DataFrame, output_dir: Path, top_n: int) -> Path:
    """Generate a forest plot for the top-N decks by reliability score."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "forest_plot.png"
    frame = deck_frame.sort_values("reliability_score", ascending=False).head(top_n).copy()

    y = np.arange(len(frame))
    means = frame["posterior_mean"].to_numpy(dtype=float)
    left = means - frame["credible_95_low"].to_numpy(dtype=float)
    right = frame["credible_95_high"].to_numpy(dtype=float) - means

    plt.figure(figsize=(12, max(6, 0.42 * len(frame))))
    plt.errorbar(means, y, xerr=[left, right], fmt="o", color="#111827", ecolor="#2563eb", capsize=4)
    labels = [f"{row.archetype} #{int(row.deck_rank_in_archetype)}" for row in frame.itertuples()]
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Posterior mean")
    plt.title("Forest plot dos decks mais confiáveis")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path
