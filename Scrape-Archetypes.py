import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://www.hsguru.com"
MIN_GAMES = 50

url = f"{BASE_URL}/meta?format=2"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0 Safari/537.36"
    )
}

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

archetypes = []

for td in soup.select("td.decklist-info"):
    a = td.select_one("a.deck-title")

    if a is None:
        continue

    name = a.text.strip()

    archetype_url = urljoin(BASE_URL, a["href"])

    decks_url = (
        f"{BASE_URL}/decks?"
        f"min_games={MIN_GAMES}"
        f"&player_deck_archetype[]={quote_plus(name)}"
    )

    archetypes.append({
        "name": name,
        "archetype_url": archetype_url,
        "decks_url": decks_url
    })

# Exibe no terminal
print(json.dumps(archetypes, indent=4, ensure_ascii=False))

# Salva em arquivo JSON
with open("archetypes.json", "w", encoding="utf-8") as f:
    json.dump(archetypes, f, indent=4, ensure_ascii=False)

print(f"\nForam encontrados {len(archetypes)} arquétipos.")
print("Arquivo salvo como 'archetypes.json'.")