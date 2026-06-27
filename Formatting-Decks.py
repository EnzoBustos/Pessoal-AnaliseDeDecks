import json
import re

INPUT_FILE = "decks.json"
OUTPUT_FILE = "analysis-ready.json"


def extract_deckcode(text: str):
    if not text:
        return None

    match = re.search(r"([A-Za-z0-9+/=]{40,})", text)
    return match.group(1) if match else None


def safe_load_json(path):
    """
    Lida com:
    - lista de dicts
    - lista de strings
    - string JSON dupla
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # caso 1: JSON veio como string inteira
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return []

    return data


def clean_decks(raw_decks):
    arquetipos = {}

    for d in raw_decks:

        # ----------------------------
        # FIX PRINCIPAL DO SEU ERRO
        # ----------------------------
        if isinstance(d, str):
            # tenta extrair direto da string
            deckcode = extract_deckcode(d)
            if not deckcode:
                continue

            archetype = "Unknown"
            winrate = None
            games = None

        else:
            archetype = d.get("archetype")
            if not archetype:
                continue

            deckcode = extract_deckcode(d.get("deckcode", ""))
            winrate = d.get("winrate")
            games = d.get("games")

        if not deckcode:
            continue

        # validação numérica
        try:
            winrate = float(winrate) if winrate is not None else None
            games = int(games) if games is not None else None
        except:
            continue

        if winrate is None or games is None:
            continue

        if archetype not in arquetipos:
            arquetipos[archetype] = []

        arquetipos[archetype].append({
            "deck_code": deckcode,
            "winrate": winrate,
            "jogos": games
        })

    return arquetipos


# -----------------------------
# LOAD
# -----------------------------
raw_decks = safe_load_json(INPUT_FILE)

# -----------------------------
# CLEAN
# -----------------------------
arquetipos = clean_decks(raw_decks)

# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(arquetipos, f, indent=4, ensure_ascii=False)

print(f"OK - Arquivo gerado: {OUTPUT_FILE}")
print(f"Arquetipos encontrados: {len(arquetipos)}")