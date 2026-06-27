"""Central configuration for the Hearthstone analysis project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from utils.constants import (
    HEADERS,
    MIN_GAMES,
    MONTE_CARLO_SAMPLES,
    PRIOR_EPS,
    PROBABILITY_THRESHOLDS,
    RANDOM_SEED,
    SCORE_K,
)
from utils.paths import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_CSV_DIR,
    OUTPUT_DIR,
    OUTPUT_FIGURES_DIR,
    OUTPUT_REPORTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved filesystem paths used by the pipeline."""

    root: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    output_dir: Path = OUTPUT_DIR
    output_csv_dir: Path = OUTPUT_CSV_DIR
    output_figures_dir: Path = OUTPUT_FIGURES_DIR
    output_reports_dir: Path = OUTPUT_REPORTS_DIR
    archetypes_file: Path = RAW_DATA_DIR / "archetypes.json"
    decks_file: Path = RAW_DATA_DIR / "decks.json"
    analysis_file: Path = RAW_DATA_DIR / "analysis-ready.json"


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Statistical and reporting parameters for the full analysis run."""

    seed: int = RANDOM_SEED
    mc_samples: int = MONTE_CARLO_SAMPLES
    games_k: float = SCORE_K
    thresholds: tuple[float, ...] = PROBABILITY_THRESHOLDS
    min_games: int = MIN_GAMES
    prior_eps: float = PRIOR_EPS
    headers: dict[str, str] = field(default_factory=lambda: dict(HEADERS))
    top_decks_global: int = 20
    top_archetypes_global: int = 10
    top_posterior_curves: int = 10
    forest_plot_size: int = 20
    paths: ProjectPaths = field(default_factory=ProjectPaths)
