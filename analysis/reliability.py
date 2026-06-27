"""Reliability score construction for decks and archetypes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reliability_score(deck_frame: pd.DataFrame, games_k: float) -> pd.DataFrame:
    """Compute the deck-level reliability score.

    score = posterior_mean * confidence * games_weight * meta_weight
    confidence = 1 - normalized_std
    games_weight = sqrt(games / (games + k))
    meta_weight = sqrt(total_games / max_total_games)
    """

    frame = deck_frame.copy()
    normalized_std = frame["posterior_std"] / float(frame["posterior_std"].max())
    frame["confidence"] = 1.0 - normalized_std.clip(lower=0.0, upper=1.0)
    frame["games_weight"] = np.sqrt(frame["jogos"] / (frame["jogos"] + games_k))

    max_total_games = float(frame["total_games"].max())
    frame["meta_weight"] = np.sqrt(frame["total_games"] / max_total_games)
    frame["reliability_score"] = (
        frame["posterior_mean"] * frame["confidence"] * frame["games_weight"] * frame["meta_weight"]
    )
    return frame
