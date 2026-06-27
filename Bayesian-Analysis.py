import json
import re
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

INPUT_FILE = "analysis-ready.json"
GRAPHS_DIR = "graficos"
VERBOSE = False


# ------------------------------------------------------------
# UTIL
# ------------------------------------------------------------
def nome_arquivo_seguro(nome):
    return re.sub(r"[^\w\-]+", "_", nome.strip())


os.makedirs(GRAPHS_DIR, exist_ok=True)


def log(message=""):
    if VERBOSE:
        print(message)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# suporta dois formatos:
# 1) {"Archetype": [...]}  (seu output atual)
# 2) lista de dicts
if isinstance(data, dict):
    arquetipos = data
else:
    arquetipos = {}
    for d in data:
        arquetipos.setdefault(d["archetype"], []).append(d)


# ------------------------------------------------------------
# BAYES CONFIG
# ------------------------------------------------------------
alpha0, beta0 = 1, 1
N_SIM = 1_000_000


# ------------------------------------------------------------
# CORE ANALYSIS
# ------------------------------------------------------------
def analisar_arquetipo(nome_arquetipo, variantes_lista):

    # normaliza estrutura
    variantes = {
        f"Deck {i}": {
            "deck_code": v["deck_code"],
            "winrate": float(v["winrate"]),
            "jogos": int(v["jogos"])
        }
        for i, v in enumerate(variantes_lista, start=1)
    }

    # binomial approximation
    for v in variantes.values():
        v["vitorias"] = round(v["winrate"] * v["jogos"])
        v["derrotas"] = v["jogos"] - v["vitorias"]

    # posterior
    posteriors = {}
    amostras = {}

    for nome, v in variantes.items():
        a = alpha0 + v["vitorias"]
        b = beta0 + v["derrotas"]

        posteriors[nome] = (a, b)
        amostras[nome] = np.random.beta(a, b, size=N_SIM)

    nomes = list(variantes.keys())
    matriz = np.vstack([amostras[n] for n in nomes])

    indice_melhor = np.argmax(matriz, axis=0)
    melhor_em_cada = np.max(matriz, axis=0)

    # resumo
    probs_melhor = {}
    metricas = {}
    for i, nome in enumerate(nomes):
        probs_melhor[nome] = np.mean(indice_melhor == i)

    for nome, v in variantes.items():
        a, b = posteriors[nome]
        media = a / (a + b)
        ic_low, ic_high = stats.beta.ppf([0.025, 0.975], a, b)
        metricas[nome] = {
            "media": media,
            "ic_low": ic_low,
            "ic_high": ic_high,
        }

    melhor_nome = max(probs_melhor, key=probs_melhor.get)
    melhor = variantes[melhor_nome]
    reliable_nome = max(metricas, key=lambda nome: metricas[nome]["ic_low"])
    reliable = variantes[reliable_nome]

    log(f"\nARQUÉTIPO: {nome_arquetipo}")
    log("RESUMO:")
    for nome, v in variantes.items():
        media = metricas[nome]["media"]
        ic_low = metricas[nome]["ic_low"]
        ic_high = metricas[nome]["ic_high"]
        log(
            f"{nome:10s} | jogos={v['jogos']:5d} | "
            f"wr={v['winrate']:.3f} | post={media:.3f} | "
            f"IC95%=[{ic_low:.3f}, {ic_high:.3f}] | "
            f"P(melhor)={probs_melhor[nome]*100:.1f}%"
        )

    prob_vencedor = probs_melhor[melhor_nome]

    if prob_vencedor >= 0.90:
        conf = "ALTA"
    elif prob_vencedor >= 0.65:
        conf = "MODERADA"
    else:
        conf = "BAIXA"

    log(f"\n>>> MELHOR: {melhor_nome}")
    log(f">>> CONF: {conf}")
    log(f">>> DECK: {melhor['deck_code']}")

    log(f">>> MAIS RELIABLE: {reliable_nome}")
    log(
        f">>> IC95% INFERIOR: {metricas[reliable_nome]['ic_low']:.3f} | "
        f"POSTERIOR: {metricas[reliable_nome]['media']:.3f}"
    )
    log(f">>> DECK RELIABLE: {reliable['deck_code']}")

    # perda esperada (reliability proxy)
    log("\nPERDA ESPERADA:")
    for i, nome in enumerate(nomes):
        loss = np.mean(melhor_em_cada - matriz[i])
        log(f"{nome:10s}: {loss*100:.2f} p.p.")

    # salvar gráfico
    x = np.linspace(0, 1, 1000)

    plt.figure(figsize=(9, 5))
    for nome, (a, b) in posteriors.items():
        plt.plot(
            x,
            stats.beta.pdf(x, a, b),
            label=f"{nome} (P={probs_melhor[nome]*100:.0f}%)"
        )

    plt.title(nome_arquetipo)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = os.path.join(GRAPHS_DIR, f"posteriors_{nome_arquivo_seguro(nome_arquetipo)}.png")
    plt.savefig(out, dpi=150)
    plt.close()

    log(f"\nGráfico: {out}")

    return {
        "arquetipo": nome_arquetipo,
        "melhor": melhor_nome,
        "deck": melhor["deck_code"],
        "mais_reliable": reliable_nome,
        "deck_reliable": reliable["deck_code"],
        "winrate_reliable": reliable["winrate"],
        "jogos_reliable": reliable["jogos"],
        "conf": conf,
        "prob": prob_vencedor,
        "ic_low_reliable": metricas[reliable_nome]["ic_low"],
        "ic_high_reliable": metricas[reliable_nome]["ic_high"],
        "posterior_reliable": metricas[reliable_nome]["media"],
    }


# ------------------------------------------------------------
# RUN ALL ARCHETYPES
# ------------------------------------------------------------
resultados = []

for nome, lista in arquetipos.items():
    resultados.append(analisar_arquetipo(nome, lista))


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------
resumo_final_path = "resumo_final.txt"

with open(resumo_final_path, "w", encoding="utf-8") as resumo_final:
    for r in resultados:
        resumo_final.write(f"{r['arquetipo']}\n")
        resumo_final.write(
            f"{r['deck_reliable']} | wr={r['winrate_reliable']:.3f} | "
            f"jogos={r['jogos_reliable']} | post={r['posterior_reliable']:.3f} | "
            f"IC95%=[{r['ic_low_reliable']:.3f}, {r['ic_high_reliable']:.3f}]\n\n"
        )

print("\nRESUMO FINAL")

for r in resultados:
    print(
        f"{r['arquetipo']}: melhor={r['melhor']} "
        f"(P={r['prob']*100:.1f}%, {r['conf']}) | "
        f"reliable={r['mais_reliable']} | "
        f"wr={r['winrate_reliable']:.3f} | jogos={r['jogos_reliable']} | "
        f"post={r['posterior_reliable']:.3f} | "
        f"IC95%=[{r['ic_low_reliable']:.3f}, {r['ic_high_reliable']:.3f}]"
    )

print(f"\nArquivo salvo em {resumo_final_path}")