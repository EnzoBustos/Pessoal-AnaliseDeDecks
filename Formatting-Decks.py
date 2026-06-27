import json

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