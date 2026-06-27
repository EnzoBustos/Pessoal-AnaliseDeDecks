import json
import os
import time
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import sys

sys.stdout.reconfigure(encoding='utf-8')

INPUT_FILE = "archetypes.json"
OUTPUT_FILE = "decks.json"


def default_state():
    return {
        "decks": [],
        "progress": {},
    }


def load_state():
    if not os.path.exists(OUTPUT_FILE):
        return default_state()

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {
            "decks": data,
            "progress": {},
        }

    if not isinstance(data, dict):
        return default_state()

    data.setdefault("decks", [])
    data.setdefault("progress", {})
    return data


def save_state(state):
    temp_file = f"{OUTPUT_FILE}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
    os.replace(temp_file, OUTPUT_FILE)


def get_archetype_progress(state, archetype_name):
    return state["progress"].setdefault(
        archetype_name,
        {
            "status": "pending",
            "saved_decks": 0,
        },
    )


def is_archetype_done(state, archetype_name):
    progress = state["progress"].get(archetype_name, {})
    return progress.get("status") == "done"


# =========================================================
# EXTRAI SOMENTE O DECKCODE BASE64
# =========================================================
def extract_deckcode(text):
    match = re.search(r"AAECA[A-Za-z0-9+/=]+", text)
    return match.group(0) if match else None


# =========================================================
# EXTRAI WINRATE + GAMES DE FORMA ROBUSTA
# =========================================================
def extract_stats(stats_block):

    winrate = None
    games = None

    if not stats_block:
        return winrate, games

    text = stats_block.get_text(" ", strip=True)

    # -------------------------
    # WINRATE (ex: 55.8)
    # -------------------------
    winrate_node = stats_block.select_one("span.tw-text-center.basic-black-text span")
    if not winrate_node and stats_block.name == "span" and "tw-text-center" in stats_block.get("class", []):
        winrate_node = stats_block.select_one("span")
    if winrate_node:
        try:
            winrate = float(winrate_node.get_text(strip=True)) / 100
        except:
            pass
    else:
        wr_match = re.search(r"(\d{1,2}\.\d)", text)
        if wr_match:
            try:
                winrate = float(wr_match.group(1)) / 100
            except:
                pass

    # -------------------------
    # GAMES (ex: 27624)
    # -------------------------
    games_node = stats_block.select_one("div.column.tag")
    if not games_node and stats_block.name == "div" and "column" in stats_block.get("class", []) and "tag" in stats_block.get("class", []):
        games_node = stats_block
    if games_node:
        games_match = re.search(r"Games:\s*([\d,]+)", games_node.get_text(" ", strip=True))
        if games_match:
            try:
                games = int(games_match.group(1).replace(",", ""))
            except:
                pass
    else:
        games_match = re.search(r"Games:\s*([\d,]+)", text)
        if games_match:
            try:
                games = int(games_match.group(1).replace(",", ""))
            except:
                pass

    return winrate, games


# =========================================================
# PARSER HTML
# =========================================================
def extract_decks(html, archetype):

    soup = BeautifulSoup(html, "html.parser")

    viewport = soup.select_one("#deck_stats_viewport")
    if not viewport:
        return []

    deck_blocks = viewport.select("div.column.is-narrow")

    decks = []

    for deck in deck_blocks:

        # =====================================================
        # DECKCODE (RAW -> CLEAN)
        # =====================================================
        deckcode = None

        button = deck.select_one("button[data-clipboard-text]")
        if button:
            raw_text = button.get("data-clipboard-text")
            deckcode = extract_deckcode(raw_text)

        if not deckcode:
            textarea = deck.select_one("textarea")
            if textarea:
                raw_text = textarea.get_text(" ", strip=True)
                deckcode = extract_deckcode(raw_text)

        if not deckcode:
            continue

        # =====================================================
        # STATS
        # =====================================================
        stats_block = deck if deck.get("id", "").startswith("deck_stats") else deck.select_one('div[id^="deck_stats"]')
        if not stats_block:
            stats_block = deck.select_one(
                "div.columns.is-multiline.is-mobile.is-text-overflow"
            )

        winrate, games = extract_stats(stats_block)

        # =====================================================
        # SAVE
        # =====================================================
        decks.append({
            "archetype": archetype["name"],
            "archetype_url": archetype["archetype_url"],
            "decks_url": archetype["decks_url"],
            "deckcode": deckcode,
            "winrate": round(winrate, 3) if winrate is not None else None,
            "games": games
        })

    return decks


# =========================================================
# MAIN
# =========================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    archetypes = json.load(f)

state = load_state()
all_decks = state["decks"]
seen_codes_by_archetype = {}

for deck in all_decks:
    archetype_name = deck.get("archetype")
    deckcode = deck.get("deckcode")
    if not archetype_name or not deckcode:
        continue
    seen_codes_by_archetype.setdefault(archetype_name, set()).add(deckcode)


with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for archetype in archetypes:

        archetype_name = archetype["name"]

        if is_archetype_done(state, archetype_name):
            print(f"\nPulando {archetype_name} (já concluído).")
            continue

        print(f"\nScraping {archetype_name}...")

        archetype_progress = get_archetype_progress(state, archetype_name)
        seen_codes = seen_codes_by_archetype.setdefault(archetype_name, set())

        page.goto(archetype["decks_url"], wait_until="networkidle")

        last_height = 0
        stable_rounds = 0
        new_decks_for_archetype = 0

        while True:

            # scroll infinito
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)

            html = page.content()
            decks = extract_decks(html, archetype)

            new_count = 0

            for d in decks:
                if d["deckcode"] not in seen_codes:
                    seen_codes.add(d["deckcode"])
                    all_decks.append(d)
                    new_count += 1
                    new_decks_for_archetype += 1

            print(f"  +{new_count} novos decks | total: {len(seen_codes)}")

            archetype_progress["status"] = "in_progress"
            archetype_progress["saved_decks"] = len(seen_codes)
            save_state(state)

            # detecção de fim do infinite scroll
            current_height = page.evaluate("document.body.scrollHeight")

            if current_height == last_height:
                stable_rounds += 1
            else:
                stable_rounds = 0

            last_height = current_height

            if stable_rounds >= 3:
                break

        archetype_progress["status"] = "done"
        archetype_progress["saved_decks"] = len(seen_codes)
        save_state(state)
        print(f"  Concluído: {new_decks_for_archetype} novos decks salvos para {archetype_name}.")

    browser.close()


# =========================================================
# OUTPUT
# =========================================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=4, ensure_ascii=False)

print(f"\nTotal final de decks: {len(all_decks)}")
print(f"Salvo em {OUTPUT_FILE}")