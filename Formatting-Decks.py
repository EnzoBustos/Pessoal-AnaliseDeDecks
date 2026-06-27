"""Legacy compatibility helper that copies raw deck data into analysis-ready format.

The new loader reads raw JSON directly, so this script is kept only as a thin
compatibility wrapper for older workflows.
"""

from __future__ import annotations

import json

from analysis.loader import load_decks
from utils.helpers import configure_utf8_console
from utils.io import save_json
from utils.paths import RAW_DATA_DIR


def main() -> None:
    """Persist the normalized raw deck list in the legacy analysis-ready format."""

    configure_utf8_console()
    decks = load_decks()
    grouped: dict[str, list[dict[str, object]]] = {}

    for _, row in decks.iterrows():
        grouped.setdefault(str(row["archetype"]), []).append(
            {
                "deck_code": row["deck_code"],
                "winrate": float(row["winrate_observed"]),
                "jogos": int(row["jogos"]),
            }
        )

    save_json(grouped, RAW_DATA_DIR / "analysis-ready.json")


if __name__ == "__main__":
    main()
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

INPUT_FILE = "decks.json"
OUTPUT_FILE = "analysis-ready.json"


def convert(raw_data):
    # suporta tanto {"decks": [...]} quanto lista direta
    if isinstance(raw_data, dict) and "decks" in raw_data:
        raw_data = raw_data["decks"]

    arquetipos = {}

    for d in raw_data:
        # segurança contra entradas quebradas
        if not isinstance(d, dict):
            continue

        archetype = d.get("archetype")
        deck_code = d.get("deckcode")
        winrate = d.get("winrate")
        games = d.get("games")

        if not archetype or not deck_code:
            continue

        # validação numérica
        try:
            winrate = float(winrate)
            games = int(games)
        except (TypeError, ValueError):
            continue

        # inicializa lista do arquétipo
        if archetype not in arquetipos:
            arquetipos[archetype] = []

        arquetipos[archetype].append({
            "deck_code": deck_code.strip(),
            "winrate": winrate,
            "jogos": games
        })

    # opcional: ordenar por "qualidade estatística" (winrate + volume)
    for archetype in arquetipos:
        arquetipos[archetype].sort(
            key=lambda x: (x["winrate"], x["jogos"]),
            reverse=True
        )

    return arquetipos


# -----------------------------
# LOAD
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# -----------------------------
# CONVERT
# -----------------------------
arquetipos = convert(raw_data)

# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(arquetipos, f, indent=4, ensure_ascii=False)

print(f"OK - Arquivo gerado: {OUTPUT_FILE}")
print(f"Arquétipos encontrados: {len(arquetipos)}")