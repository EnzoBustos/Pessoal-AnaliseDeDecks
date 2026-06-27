"""Per-archetype ranking visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARCHETYPE_RANK_FIGSIZE_MIN_HEIGHT, ARCHETYPE_RANK_FIGSIZE_ROW_HEIGHT, ARCHETYPE_RANK_FIGSIZE_WIDTH, ARCHETYPE_RANK_PLOT_SIZE, FIG_DPI, PLOT_ACCENT_COLOR, PLOT_SECONDARY_COLOR
from utils.helpers import safe_filename


def plot_archetype_rankings(deck_frame: pd.DataFrame, output_dir: Path, top_n: int = ARCHETYPE_RANK_PLOT_SIZE) -> list[Path]:
    """Generate one bar chart per archetype showing the most reliable decks.

    Each chart uses human-readable rank labels instead of dumping a long list of
    deck codes into a single global figure.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for archetype, group in deck_frame.groupby("archetype", sort=False):
        frame = group.sort_values(["reliability_score", "posterior_mean", "jogos"], ascending=[False, False, False]).head(top_n)
        if frame.empty:
            continue

        labels = [f"#{int(row.deck_rank_in_archetype)} WR {row.winrate_observed:.3f} | J {int(row.jogos)}" for row in frame.itertuples()]
        scores = frame["reliability_score"].to_numpy(dtype=float)
        means = frame["posterior_mean"].to_numpy(dtype=float)
        errors_low = means - frame["credible_95_low"].to_numpy(dtype=float)
        errors_high = frame["credible_95_high"].to_numpy(dtype=float) - means

        y = np.arange(len(frame))
        fig_height = max(ARCHETYPE_RANK_FIGSIZE_MIN_HEIGHT, ARCHETYPE_RANK_FIGSIZE_ROW_HEIGHT * len(frame) + 2)
        plt.figure(figsize=(ARCHETYPE_RANK_FIGSIZE_WIDTH, fig_height))
        plt.barh(y, scores, color=PLOT_SECONDARY_COLOR, alpha=0.85)
        plt.errorbar(means, y, xerr=[errors_low, errors_high], fmt="none", ecolor=PLOT_ACCENT_COLOR, capsize=4, linewidth=1)
        plt.yticks(y, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("Reliability Score / Posterior mean")
        plt.title(f"{archetype} - melhores decks pelas métricas")
        plt.tight_layout()

        path = output_dir / f"ranking_{safe_filename(archetype)}.png"
        plt.savefig(path, dpi=FIG_DPI)
        plt.close()
        paths.append(path)

    return paths
