"""Project path helpers based on pathlib."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_CSV_DIR = OUTPUT_DIR / "csv"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

LEGACY_ARCHETYPES_FILE = BASE_DIR / "archetypes.json"
LEGACY_DECKS_FILE = BASE_DIR / "decks.json"
LEGACY_ANALYSIS_FILE = BASE_DIR / "analysis-ready.json"


def ensure_project_directories() -> None:
    """Create the standard project folders if they do not already exist."""

    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_CSV_DIR, OUTPUT_FIGURES_DIR, OUTPUT_REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
