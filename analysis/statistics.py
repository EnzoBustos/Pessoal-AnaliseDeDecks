"""Descriptive statistics for archetypes and decks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_archetype_statistics(deck_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-archetype summary statistics."""

    rows: list[dict[str, float | int | str]] = []

    for archetype, group in deck_frame.groupby("archetype", sort=False):
        games = group["jogos"].to_numpy(dtype=float)
        shares = games / games.sum()
        posterior_means = group["posterior_mean"].to_numpy(dtype=float)

        entropy = float(stats.entropy(shares)) if len(shares) > 1 else 0.0
        normalized_entropy = entropy / float(np.log(len(shares))) if len(shares) > 1 else 0.0
        herfindahl = float(np.sum(shares**2))
        bayesian_mean = float(np.average(posterior_means, weights=games))
        posterior_std = float(np.std(posterior_means, ddof=1)) if len(posterior_means) > 1 else 0.0
        cv = posterior_std / bayesian_mean if bayesian_mean > 0 else 0.0

        idx_popular = int(group["jogos"].idxmax())
        idx_reliable = int(group["reliability_score"].idxmax())
        idx_strong = int(group["posterior_mean"].idxmax())

        rows.append(
            {
                "archetype": archetype,
                "num_decks": int(len(group)),
                "mean_observed_wr": float(group["winrate_observed"].mean()),
                "bayesian_mean": bayesian_mean,
                "median_posterior_mean": float(np.median(posterior_means)),
                "posterior_std": posterior_std,
                "cv": float(cv),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "herfindahl_index": herfindahl,
                "deck_most_popular": str(group.loc[idx_popular, "deck_code"]),
                "deck_most_reliable": str(group.loc[idx_reliable, "deck_code"]),
                "deck_most_strong": str(group.loc[idx_strong, "deck_code"]),
                "top_deck_prob_best": float(group["prob_best"].max()),
                "probability_of_superior_deck": float(1.0 - group["prob_best"].max()),
                "total_games": int(group["total_games"].iloc[0]),
            }
        )

    return pd.DataFrame(rows)
