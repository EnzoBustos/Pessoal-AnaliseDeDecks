"""Input loading and normalization for the Bayesian analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.helpers import read_if_exists
from utils.io import load_json
from utils.paths import LEGACY_ANALYSIS_FILE, LEGACY_ARCHETYPES_FILE, LEGACY_DECKS_FILE, RAW_DATA_DIR


def _load_json_candidates(*candidates: Path) -> Any:
    """Load the first existing JSON candidate."""

    path = read_if_exists(*candidates)
    if path is None:
        raise FileNotFoundError(f"Nenhum dos arquivos foi encontrado: {candidates}")
    return load_json(path)


def load_archetypes() -> pd.DataFrame:
    """Load archetype metadata from raw or legacy JSON files."""

    data = _load_json_candidates(RAW_DATA_DIR / "archetypes.json", LEGACY_ARCHETYPES_FILE)
    if isinstance(data, list):
        frame = pd.DataFrame(data)
    else:
        frame = pd.DataFrame.from_dict(data, orient="index").reset_index(names="name")

    if "name" not in frame.columns or "total_games" not in frame.columns:
        raise ValueError("archetypes.json precisa conter as colunas name e total_games.")

    frame = frame[["name", "total_games"]].copy()
    frame["total_games"] = frame["total_games"].fillna(0).astype(int)
    return frame


def load_decks() -> pd.DataFrame:
    """Load deck observations from raw or legacy JSON files."""

    data = _load_json_candidates(RAW_DATA_DIR / "decks.json", LEGACY_DECKS_FILE, LEGACY_ANALYSIS_FILE)
    if isinstance(data, dict) and "decks" in data:
        data = data["decks"]
    if not isinstance(data, list):
        raise ValueError("decks.json deve conter uma lista de decks.")

    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        archetype = item.get("archetype")
        deck_code = item.get("deck_code") or item.get("deckcode")
        winrate = item.get("winrate")
        games = item.get("jogos") if item.get("jogos") is not None else item.get("games")
        if archetype is None or deck_code is None or winrate is None or games is None:
            continue

        rows.append(
            {
                "archetype": str(archetype),
                "deck_code": str(deck_code),
                "winrate_observed": float(winrate),
                "jogos": int(games),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Nenhum deck válido foi encontrado.")

    return frame


def prepare_dataset() -> pd.DataFrame:
    """Join archetype metadata and deck observations into a normalized dataset."""

    archetypes = load_archetypes()
    decks = load_decks()

    frame = decks.merge(archetypes, how="left", left_on="archetype", right_on="name")
    frame.drop(columns=["name"], inplace=True)
    frame["total_games"] = frame["total_games"].fillna(frame.groupby("archetype")["jogos"].transform("sum"))
    frame["total_games"] = frame["total_games"].astype(int)
    frame["wins"] = np.rint(frame["winrate_observed"] * frame["jogos"]).astype(int)
    frame["losses"] = frame["jogos"] - frame["wins"]
    return frame
