"""Deck-code extraction helpers reused by the deck scraper."""

from __future__ import annotations

import re


def extract_deckcode(text: str) -> str | None:
    """Return the first valid deck code found in a text blob."""

    match = re.search(r"AAECA[A-Za-z0-9+/=]+", text)
    return match.group(0) if match else None
