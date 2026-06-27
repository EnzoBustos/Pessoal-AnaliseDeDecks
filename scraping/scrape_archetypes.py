"""Scrape archetype metadata from HSGuru and save it to raw JSON."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote_plus

from tqdm import tqdm

from config import ARCHETYPE_META_PATH, BASE_URL, MIN_GAMES
from scraping.scraper_utils import download_page, parse_table, resolve_url, save_json_file
from utils.helpers import configure_utf8_console
from utils.paths import RAW_DATA_DIR


LOGGER = logging.getLogger(__name__)


def scrape_archetypes(output_path: Path | None = None) -> list[dict[str, object]]:
    """Scrape all archetypes from the HSGuru meta page."""

    target_path = output_path or (RAW_DATA_DIR / "archetypes.json")
    html = download_page(f"{BASE_URL}{ARCHETYPE_META_PATH}")
    soup = parse_table(html)

    archetypes: list[dict[str, object]] = []
    rows = soup.select("td.decklist-info")
    for row in tqdm(rows, desc="Arquétipos", unit="arquétipo"):
        link = row.select_one("a.deck-title")
        if link is None:
            continue

        name = link.get_text(strip=True)
        popularity_cell = row.parent.select_one("td:nth-of-type(3)") if row.parent else None
        total_games = None
        if popularity_cell is not None:
            match = re.search(r"\((\d+)\)", popularity_cell.get_text(strip=True))
            if match is not None:
                total_games = int(match.group(1))

        archetypes.append(
            {
                "name": name,
                "archetype_url": resolve_url(BASE_URL, link["href"]),
                "decks_url": f"{BASE_URL}/decks?min_games={MIN_GAMES}&player_deck_archetype[]={quote_plus(name)}",
                "total_games": total_games,
            }
        )

    save_json_file(archetypes, target_path)
    LOGGER.info("%s arquétipos salvos em %s", len(archetypes), target_path)
    return archetypes


def main() -> None:
    """CLI entry point."""

    configure_utf8_console()
    scrape_archetypes()


if __name__ == "__main__":
    main()
