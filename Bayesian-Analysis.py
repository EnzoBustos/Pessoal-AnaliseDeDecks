"""Legacy compatibility entry point for the Bayesian analysis pipeline."""

from utils.helpers import configure_utf8_console

from main import main


if __name__ == "__main__":
    configure_utf8_console()
    main()
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
np.random.seed(42)

INPUT_FILE = "analysis-ready.json"
ARCHETYPE_FILE = "archetypes.json"
GRAPHS_DIR = "graficos"

VERBOSE = False
MAKE_GRAPHS = True

ALPHA0 = 1.0
BETA0 = 1.0
N_SIM = 200_000

SCORE_BASE_JOGOS = 10.0
SCORE_PESO_JOGOS = 1.5
SCORE_PESO_TOTAL_GAMES = 0.75
SCORE_EPS_LARGURA = 1e-6

SENSIBILITY_JOGOS = [1.0, 1.5, 2.0]
SENSIBILITY_TOTAL = [0.5, 0.75, 1.0]


def nome_arquivo_seguro(nome):
    return re.sub(r"[^\w\-]+", "_", nome.strip())


def log(message=""):
    if VERBOSE:
        print(message)


def carregar_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_arquetipos():
    data = carregar_json(INPUT_FILE)

    if isinstance(data, dict):
        arquetipos = data
    else:
        arquetipos = {}
        for item in data:
            arquetipos.setdefault(item["archetype"], []).append(item)

    meta = {}
    if os.path.exists(ARCHETYPE_FILE):
        meta_data = carregar_json(ARCHETYPE_FILE)
        if isinstance(meta_data, list):
            meta = {item["name"]: item for item in meta_data if isinstance(item, dict) and "name" in item}
        elif isinstance(meta_data, dict):
            meta = meta_data

    return arquetipos, meta


def total_games_do_arquetipo(nome_arquetipo, variantes_lista, meta_arquetipos):
    meta = meta_arquetipos.get(nome_arquetipo, {})
    if isinstance(meta, dict) and meta.get("total_games") is not None:
        return int(meta["total_games"])
    return int(sum(int(v["jogos"]) for v in variantes_lista))


def score_deck(posterior_media, ic_width, jogos, total_games_arquetipo):
    edge = posterior_media - 0.5
    support_deck = np.log10(jogos + SCORE_BASE_JOGOS) ** SCORE_PESO_JOGOS
    support_total = np.log10(total_games_arquetipo + SCORE_BASE_JOGOS) ** SCORE_PESO_TOTAL_GAMES
    precision = 1.0 / (ic_width + SCORE_EPS_LARGURA)
    return edge * support_deck * support_total * precision


def analisar_arquetipo(nome_arquetipo, variantes_lista, total_games_arquetipo):
    variantes = {
        f"Deck {i}": {
            "deck_code": v["deck_code"],
            "winrate": float(v["winrate"]),
            "jogos": int(v["jogos"]),
        }
        for i, v in enumerate(variantes_lista, start=1)
    }

    for v in variantes.values():
        v["vitorias"] = round(v["winrate"] * v["jogos"])
        v["derrotas"] = v["jogos"] - v["vitorias"]

    posteriors = {}
    amostras = {}
    metricas = {}

    for nome, v in variantes.items():
        a = ALPHA0 + v["vitorias"]
        b = BETA0 + v["derrotas"]
        posteriors[nome] = (a, b)
        amostras[nome] = np.random.beta(a, b, size=N_SIM)

    nomes = list(variantes.keys())
    matriz = np.vstack([amostras[n] for n in nomes])
    indice_melhor = np.argmax(matriz, axis=0)

    probs_melhor = {nome: float(np.mean(indice_melhor == i)) for i, nome in enumerate(nomes)}

    for nome, v in variantes.items():
        a, b = posteriors[nome]
        posterior_media = a / (a + b)
        ic_low, ic_high = stats.beta.ppf([0.025, 0.975], a, b)
        ic_width = ic_high - ic_low
        edge = posterior_media - 0.5
        score = score_deck(posterior_media, ic_width, v["jogos"], total_games_arquetipo)

        metricas[nome] = {
            "posterior_media": posterior_media,
            "ic_low": ic_low,
            "ic_high": ic_high,
            "ic_width": ic_width,
            "edge": edge,
            "score": score,
            "prob_best": probs_melhor[nome],
        }

    ranking_decks = sorted(nomes, key=lambda nome: metricas[nome]["score"], reverse=True)
    best_name = ranking_decks[0]
    best_metric = metricas[best_name]

    top_k = ranking_decks[: min(3, len(ranking_decks))]
    archetype_score = float(np.mean([metricas[nome]["score"] for nome in top_k]))
    archetype_depth_score = float(np.mean([metricas[nome]["posterior_media"] for nome in top_k]))
    archetype_support = np.log10(total_games_arquetipo + SCORE_BASE_JOGOS)
    best_prob = best_metric["prob_best"]

    if best_prob >= 0.90:
        conf = "ALTA"
    elif best_prob >= 0.65:
        conf = "MODERADA"
    else:
        conf = "BAIXA"

    return {
        "arquetipo": nome_arquetipo,
        "total_games": total_games_arquetipo,
        "variantes": variantes,
        "metricas": metricas,
        "ranking_decks": ranking_decks,
        "deck_reliable": best_name,
        "deck_code_reliable": variantes[best_name]["deck_code"],
        "winrate_reliable": variantes[best_name]["winrate"],
        "jogos_reliable": variantes[best_name]["jogos"],
        "posterior_reliable": best_metric["posterior_media"],
        "ic_low_reliable": best_metric["ic_low"],
        "ic_high_reliable": best_metric["ic_high"],
        "ic_width_reliable": best_metric["ic_width"],
        "score_reliable": best_metric["score"],
        "prob_reliable": best_prob,
        "conf": conf,
        "archetype_score": archetype_score,
        "archetype_depth_score": archetype_depth_score,
        "archetype_support": archetype_support,
    }


def gerar_grafico_arquetipo(resultado):
    nome_arquetipo = resultado["arquetipo"]
    ranking = resultado["ranking_decks"][:10]
    metricas = resultado["metricas"]
    variantes = resultado["variantes"]

    labels = []
    medias = []
    erros_baixo = []
    erros_alto = []
    jogos = []

    for nome in ranking:
        labels.append(nome)
        medias.append(metricas[nome]["posterior_media"])
        erros_baixo.append(metricas[nome]["posterior_media"] - metricas[nome]["ic_low"])
        erros_alto.append(metricas[nome]["ic_high"] - metricas[nome]["posterior_media"])
        jogos.append(variantes[nome]["jogos"])

    y = np.arange(len(labels))

    plt.figure(figsize=(11, 6))
    plt.barh(y, medias, color="#2F6BFF", alpha=0.85)
    plt.errorbar(
        medias,
        y,
        xerr=[erros_baixo, erros_alto],
        fmt="none",
        ecolor="#1C1C1C",
        capsize=3,
        linewidth=1,
    )
    plt.yticks(y, [f"{label} ({jogos[idx]} jogos)" for idx, label in enumerate(labels)])
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.xlabel("Winrate posterior")
    plt.title(f"{nome_arquetipo} - decks mais fortes e intervalo de confiança")
    plt.tight_layout()

    out = os.path.join(GRAPHS_DIR, f"arquetipo_{nome_arquivo_seguro(nome_arquetipo)}.png")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def gerar_grafico_ranking_global(resultados_ordenados):
    labels = [r["arquetipo"] for r in resultados_ordenados]
    scores = [r["archetype_score"] for r in resultados_ordenados]
    total_games = [r["total_games"] for r in resultados_ordenados]

    y = np.arange(len(labels))
    cores = plt.cm.Blues(np.linspace(0.35, 0.9, len(labels)))

    plt.figure(figsize=(12, max(6, 0.32 * len(labels))))
    plt.barh(y, scores, color=cores)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Score robusto do arquétipo")
    plt.title("Ranking global dos arquétipos")

    for i, (score, games) in enumerate(zip(scores, total_games)):
        plt.text(score, i, f"  {games}", va="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "ranking_global_arquetipos.png")
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def gerar_grafico_sensibilidade(resultados):
    nomes = [r["arquetipo"] for r in resultados]
    contagem = defaultdict(int)
    total_cenarios = 0

    for peso_jogos in SENSIBILITY_JOGOS:
        for peso_total in SENSIBILITY_TOTAL:
            total_cenarios += 1
            scores = []
            for resultado in resultados:
                best_name = resultado["deck_reliable"]
                metric = resultado["metricas"][best_name]
                jogos = resultado["jogos_reliable"]
                total_games = resultado["total_games"]
                edge = metric["posterior_media"] - 0.5
                support_deck = np.log10(jogos + SCORE_BASE_JOGOS) ** peso_jogos
                support_total = np.log10(total_games + SCORE_BASE_JOGOS) ** peso_total
                precision = 1.0 / (metric["ic_width"] + SCORE_EPS_LARGURA)
                scores.append(edge * support_deck * support_total * precision)

            vencedor = nomes[int(np.argmax(scores))]
            contagem[vencedor] += 1

    ordenado = sorted(contagem.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ordenado]
    valores = [item[1] / total_cenarios for item in ordenado]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, valores, color="#1F8A70")
    plt.ylim(0, 1)
    plt.ylabel("Frequência como #1 em cenários de sensibilidade")
    plt.xticks(rotation=30, ha="right")
    plt.title("Robustez do ranking sob variação de pesos")
    plt.tight_layout()

    out = os.path.join(GRAPHS_DIR, "sensibilidade_ranking.png")
    plt.savefig(out, dpi=160)
    plt.close()

    return out, contagem, total_cenarios


def salvar_resumo(resultados_ordenados, sensibilidade, total_cenarios):
    resumo_final_path = "resumo_final.txt"
    contagem_sensibilidade = sensibilidade

    with open(resumo_final_path, "w", encoding="utf-8") as resumo_final:
        resumo_final.write("Ranking robusto dos arquétipos\n\n")
        for r in resultados_ordenados:
            resumo_final.write(
                f"{r['arquetipo']}\n"
                f"score={r['archetype_score']:.6f} | depth={r['archetype_depth_score']:.3f} | "
                f"support={r['archetype_support']:.3f} | total_games={r['total_games']}\n"
                f"deck={r['deck_code_reliable']} | wr={r['winrate_reliable']:.3f} | jogos={r['jogos_reliable']} | "
                f"post={r['posterior_reliable']:.3f} | IC95%=[{r['ic_low_reliable']:.3f}, {r['ic_high_reliable']:.3f}] | "
                f"P(best deck)={r['prob_reliable']*100:.1f}%\n\n"
            )

        resumo_final.write("Robustez por sensibilidade\n")
        for nome, contagem in sorted(contagem_sensibilidade.items(), key=lambda item: item[1], reverse=True):
            resumo_final.write(f"{nome}: {contagem}/{total_cenarios}\n")

    return resumo_final_path


arquetipos, meta_arquetipos = carregar_arquetipos()
os.makedirs(GRAPHS_DIR, exist_ok=True)

resultados = []
for nome, lista in arquetipos.items():
    total_games = total_games_do_arquetipo(nome, lista, meta_arquetipos)
    resultados.append(analisar_arquetipo(nome, lista, total_games))

resultados_ordenados = sorted(resultados, key=lambda r: r["archetype_score"], reverse=True)

grafico_ranking = None
grafico_sensibilidade = None
contagem_sensibilidade = defaultdict(int)
total_cenarios = 0

if MAKE_GRAPHS:
    for resultado in resultados_ordenados:
        gerar_grafico_arquetipo(resultado)

    grafico_ranking = gerar_grafico_ranking_global(resultados_ordenados)
    grafico_sensibilidade, contagem_sensibilidade, total_cenarios = gerar_grafico_sensibilidade(resultados_ordenados)

resumo_final_path = salvar_resumo(resultados_ordenados, contagem_sensibilidade, total_cenarios)

melhor = resultados_ordenados[0]
segundo = resultados_ordenados[1] if len(resultados_ordenados) > 1 else None
freq_melhor = (contagem_sensibilidade.get(melhor["arquetipo"], 0) / total_cenarios) if total_cenarios else 0.0

print("\nRESUMO FINAL")
for r in resultados_ordenados:
    print(
        f"{r['arquetipo']}: score={r['archetype_score']:.3f} | "
        f"deck={r['deck_reliable']} | wr={r['winrate_reliable']:.3f} | jogos={r['jogos_reliable']} | "
        f"total_games={r['total_games']} | P(best deck)={r['prob_reliable']*100:.1f}% | "
        f"IC95%=[{r['ic_low_reliable']:.3f}, {r['ic_high_reliable']:.3f}]"
    )

print(
    f"\nMelhor arquétipo robusto: {melhor['arquetipo']} | "
    f"score={melhor['archetype_score']:.3f} | deck={melhor['deck_reliable']} | "
    f"P(best deck)={melhor['prob_reliable']*100:.1f}% | total_games={melhor['total_games']}"
)

if segundo is not None:
    gap = melhor["archetype_score"] - segundo["archetype_score"]
    print(f"Gap para o segundo colocado: {gap:.3f}")

if total_cenarios:
    print(f"Robustez de sensibilidade do vencedor: {freq_melhor*100:.1f}% dos cenários")

if grafico_ranking is not None:
    print(f"Gráfico global salvo em {grafico_ranking}")
if grafico_sensibilidade is not None:
    print(f"Gráfico de sensibilidade salvo em {grafico_sensibilidade}")

print(f"\nArquivo salvo em {resumo_final_path}")