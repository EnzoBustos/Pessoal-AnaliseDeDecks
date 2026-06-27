"""Deck and archetype ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_deck_ranking(deck_frame: pd.DataFrame) -> pd.DataFrame:
    """Rank all decks by reliability score."""

    ranking = deck_frame.sort_values(
        ["reliability_score", "posterior_mean", "jogos"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "global_rank", np.arange(1, len(ranking) + 1))
    ranking["deck_rank_in_archetype"] = ranking.groupby("archetype").cumcount() + 1
    return ranking


def build_archetype_ranking(deck_frame: pd.DataFrame, archetype_stats: pd.DataFrame) -> pd.DataFrame:
    """Rank archetypes using the weighted average of the top deck scores."""

    max_total_games = float(deck_frame["total_games"].max())

    rows: list[dict[str, float | int | str]] = []
    for archetype, group in deck_frame.groupby("archetype", sort=False):
        top_decks = group.sort_values("reliability_score", ascending=False).head(3)
        weights = np.clip(top_decks["prob_best"].to_numpy(dtype=float), 1e-9, None)
        mean_top_score = float(np.average(top_decks["reliability_score"].to_numpy(dtype=float), weights=weights))
        support_weight = float(np.sqrt(group["total_games"].iloc[0] / max_total_games))
        archetype_score = mean_top_score * support_weight
        best_deck_idx = int(group["reliability_score"].idxmax())

        rows.append(
            {
                "archetype": archetype,
                "best_deck_code": str(group.loc[best_deck_idx, "deck_code"]),
                "best_deck_score": float(group.loc[best_deck_idx, "reliability_score"]),
                "mean_top3_score": mean_top_score,
                "support_weight": support_weight,
                "archetype_reliability_score": archetype_score,
            }
        )

    ranking = pd.DataFrame(rows).merge(archetype_stats, on="archetype", how="left")
    ranking = ranking.sort_values(
        ["archetype_reliability_score", "best_deck_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "global_rank", np.arange(1, len(ranking) + 1))
    return ranking


def build_archetype_top_decks(deck_frame: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return the top decks within each archetype ordered by reliability score."""

    rows: list[dict[str, float | int | str]] = []

    for archetype, group in deck_frame.groupby("archetype", sort=False):
        ordered = group.sort_values(["reliability_score", "posterior_mean", "jogos"], ascending=[False, False, False]).head(top_n)
        for position, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "archetype": archetype,
                    "archetype_rank": position,
                    "deck_code": str(row["deck_code"]),
                    "winrate_observed": float(row["winrate_observed"]),
                    "jogos": int(row["jogos"]),
                    "posterior_mean": float(row["posterior_mean"]),
                    "posterior_std": float(row["posterior_std"]),
                    "prob_best": float(row["prob_best"]),
                    "reliability_score": float(row["reliability_score"]),
                    "deck_rank_in_archetype": int(row["deck_rank_in_archetype"]),
                }
            )

    return pd.DataFrame(rows)
