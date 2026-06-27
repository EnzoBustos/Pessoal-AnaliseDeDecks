"""Central configuration for the Hearthstone analysis project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_CSV_DIR = OUTPUT_DIR / "csv"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

BASE_URL = "https://www.hsguru.com"
ARCHETYPE_META_PATH = "/meta?format=2"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/137.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

MIN_GAMES = 50
RANDOM_SEED = 42
MONTE_CARLO_SAMPLES = 1_000_000
PRIOR_EPS = 1e-9
PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60)
SCORE_K = 100.0

PRIOR_METHOD = "empirical_bayes"
PRIOR_BOUNDS = (-12.0, 12.0)
PRIOR_START_FLOOR = 1e-3
METHOD_OF_MOMENTS_VARIANCE_FLOOR = 1e-6

REQUEST_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30
RETRY_BACKOFF_SECONDS = 1.0
HEADLESS_BROWSER = True
NETWORK_IDLE_WAIT_UNTIL = "networkidle"

SCROLL_WAIT_SECONDS = 1.5
SCROLL_STABLE_ROUNDS = 2

TOP_DECKS_GLOBAL = 20
TOP_ARCHETYPES_GLOBAL = 10
TOP_POSTERIOR_CURVES = 10
FOREST_PLOT_SIZE = 20
ARCHETYPE_RANK_PLOT_SIZE = 5
ARCHETYPE_SUMMARY_TOP_N = 3

SCATTER_FIGSIZE = (11, 7)
FOREST_FIGSIZE_WIDTH = 12
FOREST_FIGSIZE_MIN_HEIGHT = 6
FOREST_FIGSIZE_ROW_HEIGHT = 0.42
POSTERIOR_FIGSIZE = (12, 7)
SHRINKAGE_FIGSIZE = (13, 7)
PROBABILITY_FIGSIZE = (12, 6)
HEATMAP_FIGSIZE_WIDTH = 12
HEATMAP_FIGSIZE_MIN_HEIGHT = 5
HEATMAP_FIGSIZE_ROW_HEIGHT = 0.42
ARCHETYPE_RANK_FIGSIZE_WIDTH = 11
ARCHETYPE_RANK_FIGSIZE_MIN_HEIGHT = 4.5
ARCHETYPE_RANK_FIGSIZE_ROW_HEIGHT = 0.55

FIG_DPI = 180
SCATTER_CMAP = "viridis"
HEATMAP_CMAP = "YlGnBu"
SCATTER_COLORBAR_LABEL = "Reliability Score"
PLOT_SCATTER_COLOR = "#2563eb"
PLOT_SECONDARY_COLOR = "#111827"
PLOT_ACCENT_COLOR = "#f59e0b"

SUMMARY_FILE_NAME = "resumo_final.txt"
REPORT_FILE_NAME = "report.html"
ANALYSIS_CSV_NAME = "analysis.csv"
RANKING_CSV_NAME = "ranking.csv"
DECK_STATS_CSV_NAME = "deck_statistics.csv"
ARCHETYPE_STATS_CSV_NAME = "archetype_statistics.csv"
ARCHETYPE_TOP_DECKS_CSV_NAME = "archetype_top_decks.csv"

TERMINAL_ENCODING = "utf-8"

LOG_LEVEL_NAME = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


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
    top_decks_global: int = TOP_DECKS_GLOBAL
    top_archetypes_global: int = TOP_ARCHETYPES_GLOBAL
    top_posterior_curves: int = TOP_POSTERIOR_CURVES
    forest_plot_size: int = FOREST_PLOT_SIZE
    archetype_rank_plot_size: int = ARCHETYPE_RANK_PLOT_SIZE
    archetype_summary_top_n: int = ARCHETYPE_SUMMARY_TOP_N
    paths: ProjectPaths = field(default_factory=ProjectPaths)
