"""Project-wide constants."""

from __future__ import annotations

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0 Safari/537.36"
    )
}

MIN_GAMES = 50
RANDOM_SEED = 42
MONTE_CARLO_SAMPLES = 100_000
PRIOR_EPS = 1e-9
PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60)
SCORE_K = 100.0
