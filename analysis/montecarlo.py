"""Vectorized Monte Carlo simulation for best-deck probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


def probability_of_best(deck_frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    """Estimate the probability of each deck being the best in its archetype."""

    rng = np.random.default_rng(seed)
    frame = deck_frame.copy()
    probabilities = np.zeros(len(frame), dtype=float)

    groups = list(frame.groupby("archetype", sort=False))
    for archetype, group in tqdm(groups, desc="Monte Carlo", unit="arquétipo"):
        alpha = group["posterior_alpha"].to_numpy(dtype=float)
        beta = group["posterior_beta"].to_numpy(dtype=float)

        draws = rng.beta(alpha[:, None], beta[:, None], size=(len(group), samples))
        winners = np.argmax(draws, axis=0)
        counts = np.bincount(winners, minlength=len(group))
        probabilities[group.index.to_numpy()] = counts / float(samples)

    frame["prob_best"] = probabilities
    return frame
