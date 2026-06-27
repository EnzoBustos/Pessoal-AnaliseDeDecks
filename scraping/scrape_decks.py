"""Scrape deck lists for each archetype and persist the result to raw JSON."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from config import HEADLESS_BROWSER, NETWORK_IDLE_WAIT_UNTIL, SCROLL_STABLE_ROUNDS, SCROLL_WAIT_SECONDS
from scraping.scrape_deckcodes import extract_deckcode
from scraping.scraper_utils import save_json_file
from utils.helpers import configure_utf8_console
from utils.io import load_json
from utils.paths import RAW_DATA_DIR


LOGGER = logging.getLogger(__name__)
INPUT_FILE = RAW_DATA_DIR / "archetypes.json"
OUTPUT_FILE = RAW_DATA_DIR / "decks.json"


def _load_archetypes(path: Path = INPUT_FILE) -> list[dict[str, object]]:
    """Load archetype metadata from raw JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("archetypes.json deve conter uma lista de arquétipos.")
    return [item for item in data if isinstance(item, dict)]


def _extract_decks(html: str, archetype: dict[str, object]) -> list[dict[str, object]]:
    """Extract deck records from the current page HTML."""

    soup = BeautifulSoup(html, "html.parser")
    viewport = soup.select_one("#deck_stats_viewport")
    if viewport is None:
        return []

    decks: list[dict[str, object]] = []
    for block in viewport.select("div.column.is-narrow"):
        deckcode = None

        button = block.select_one("button[data-clipboard-text]")
        if button is not None:
            deckcode = extract_deckcode(button.get("data-clipboard-text", ""))

        if deckcode is None:
            textarea = block.select_one("textarea")
            if textarea is not None:
                deckcode = extract_deckcode(textarea.get_text(" ", strip=True))

        if deckcode is None:
            continue

        stats_block = block if str(block.get("id", "")).startswith("deck_stats") else block.select_one('div[id^="deck_stats"]')
        stats_block = stats_block or block.select_one("div.columns.is-multiline.is-mobile.is-text-overflow")
        if stats_block is None:
            continue

        text = stats_block.get_text(" ", strip=True)
        winrate = None
        games = None

        winrate_node = stats_block.select_one("span.tw-text-center.basic-black-text span")
        if winrate_node is not None:
            try:
                winrate = float(winrate_node.get_text(strip=True)) / 100.0
            except ValueError:
                winrate = None

        if winrate is None:
            match = re.search(r"(\d{1,2}\.\d)", text)
            if match is not None:
                winrate = float(match.group(1)) / 100.0

        if games is None:
            match = re.search(r"Games:\s*([\d,]+)", text)
            if match is not None:
                games = int(match.group(1).replace(",", ""))

        decks.append(
            {
                "archetype": archetype["name"],
                "archetype_url": archetype["archetype_url"],
                "decks_url": archetype["decks_url"],
                "deckcode": deckcode,
                "winrate": round(winrate, 3) if winrate is not None else None,
                "games": games,
            }
        )

    return decks


def scrape_decks(input_path: Path = INPUT_FILE, output_path: Path = OUTPUT_FILE) -> list[dict[str, object]]:
    """Scrape all decks for all archetypes and save them as JSON."""

    archetypes = _load_archetypes(input_path)
    all_decks: list[dict[str, object]] = []
    seen: dict[str, set[str]] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS_BROWSER)
        page = browser.new_page()

        for archetype in tqdm(archetypes, desc="Arquétipos", unit="arquétipo"):
            archetype_name = str(archetype["name"])
            seen.setdefault(archetype_name, set())

            LOGGER.info("Scraping %s", archetype_name)
            page.goto(str(archetype["decks_url"]), wait_until=NETWORK_IDLE_WAIT_UNTIL)

            last_height = 0
            stable_rounds = 0

            scroll_bar = tqdm(desc=f"{archetype_name}", unit="scroll", leave=False)
            while True:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(SCROLL_WAIT_SECONDS)

                decks = _extract_decks(page.content(), archetype)
                new_count = 0
                for deck in decks:
                    deckcode = str(deck["deckcode"])
                    if deckcode in seen[archetype_name]:
                        continue
                    seen[archetype_name].add(deckcode)
                    all_decks.append(deck)
                    new_count += 1

                current_height = page.evaluate("document.body.scrollHeight")
                if current_height == last_height:
                    stable_rounds += 1
                else:
                    stable_rounds = 0

                last_height = current_height
                if stable_rounds >= SCROLL_STABLE_ROUNDS:
                    scroll_bar.close()
                    break

                LOGGER.info("%s | +%s novos decks | total=%s", archetype_name, new_count, len(seen[archetype_name]))
                scroll_bar.update(1)

        browser.close()

    save_json_file(all_decks, output_path)
    LOGGER.info("%s decks salvos em %s", len(all_decks), output_path)
    return all_decks


def main() -> None:
    """CLI entry point."""

    configure_utf8_console()
    scrape_decks()


if __name__ == "__main__":
    main()
