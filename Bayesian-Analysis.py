"""
Análise Bayesiana de Eficiência de Variantes de Deck (Modelo Beta-Binomial)
============================================================================

MODELO
------
Cada variante i tem um winrate "verdadeiro" desconhecido p_i. Cada partida
é um ensaio de Bernoulli(p_i), logo o número de vitórias W_i, dado n_i
partidas, segue uma distribuição Binomial(n_i, p_i).

Colocamos um prior Beta(alpha0, beta0) sobre p_i. Como a Beta é conjugada
da Binomial, a posterior também é Beta, em forma fechada:

    p_i | dados  ~  Beta(alpha0 + vitorias_i, beta0 + derrotas_i)

Isso dá, para cada variante, uma distribuição COMPLETA sobre o quão boa
ela realmente é (não só um número pontual), e permite responder de forma
probabilística:

    P(variante A é melhor que variante B | dados)
    P(variante X é a melhor de todas | dados)
    Perda esperada ao escolher cada variante

MÚLTIPLOS ARQUÉTIPOS
---------------------
Cada arquétipo (ex: "Burn Mage", "Dude Paladin") roda sua PRÓPRIA análise,
de forma totalmente independente. Isso é proposital: arquétipos diferentes
enfrentam metas de matchup diferentes, então "qual a melhor variante" só
faz sentido DENTRO de um mesmo arquétipo - nunca comparando winrate de um
arquétipo contra outro. A função `analisar_arquetipo` roda o modelo do zero
para cada um, e o loop principal só repete o processo arquétipo por
arquétipo.

COMO USAR
---------
Edite o dicionário `arquetipos` na seção 1: uma entrada por arquétipo,
cada uma com a lista de variantes (deck_code, winrate, jogos). Rode o
script - ele imprime os resultados de cada arquétipo e salva um gráfico
de posteriors por arquétipo (ex: posteriors_Burn_Mage.png).
"""

import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)  # reprodutibilidade da simulação Monte Carlo

# ---------------------------------------------------------------------------
# 1) DADOS DE ENTRADA - MÚLTIPLOS ARQUÉTIPOS
# ---------------------------------------------------------------------------
# Uma chave por arquétipo (usada só pra organizar a saída e nomear os
# gráficos). O valor é a lista de variantes daquele arquétipo, cada uma com:
#   - deck_code: o código de importação do Hearthstone
#   - winrate: em decimal (0 a 1)
#   - jogos: total de partidas
#
# Não precisa nomear as variantes dentro do arquétipo - o deck_code já
# identifica qual é qual. Rótulos "Deck 1", "Deck 2"... são gerados
# automaticamente só para as tabelas.
#
# OBS: como winrate normalmente já vem arredondado, vitorias = round(winrate
# * jogos) é uma aproximação. Se você tiver o número exato de vitórias/
# derrotas, troque por esses valores diretamente.

arquetipos = {
    "Harold Rogue": [
        {"deck_code": "AAECAaIHDpGfBKGBB5KDB4KYB9GdB+ylB4aoB4eoB4ioB9C/B/bJB4rUB5vUB4jZBwj3nwSQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.569, "jogos": 1074},
        {"deck_code": "AAECAaIHCqGBB5KDB4KYB+ylB9C/B/bJB4rUB5vUB4jZB8PyBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.568, "jogos": 7454},
        {"deck_code": "AAECAaIHCsODB9GdB+ylB4aoB4eoB4ioB9C/B4rUB5vUB4jZBwr3nwT3gQeQgweMrQfHrgfZrweaswe0wQedxQfVxQcAAA==", "winrate": 0.567, "jogos": 1322},
        {"deck_code": "AAECAaIHCqGBB5KDB8ODB4KYB+ylB9C/B/bJB4rUB5vUB4jZBwqRnwT3nwSQgwfBlwfHrge0wQfAwQedxQfVxQfD8gcAAA==", "winrate": 0.566, "jogos": 1734},
        {"deck_code": "AAECAaIHCqGBB4KYB+ylB4++B9C/B/bJB4rUB5vUB4jZB8PyBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.565, "jogos": 2165},
        {"deck_code": "AAECAaIHCqGBB5KDB8ODB4KYB+ylB9C/B/bJB4rUB5vUB4jZBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.565, "jogos": 33190},
        {"deck_code": "AAECAaIHDqGBB5KDB8GXB4KYB9GdB+ylB4aoB4eoB4ioB9C/B/bJB4rUB5vUB4jZBwiRnwT3nwSQgwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.563, "jogos": 6164},
        {"deck_code": "AAECAaIHDKGBB5KDB4KYB+ylB4aoB4eoB4ioB9C/B/bJB4rUB5vUB4jZBwmRnwT3nwSQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.563, "jogos": 10150},
        {"deck_code": "AAECAaIHCtmiBqGBB5KDB4KYB+ylB9C/B/bJB4rUB5vUB4jZBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.563, "jogos": 20724},
    ],
}

# ---------------------------------------------------------------------------
# 2) PRIOR e CONFIGURAÇÃO DA SIMULAÇÃO
# ---------------------------------------------------------------------------
# Beta(1,1) = uniforme em [0,1]: "antes dos dados, qualquer winrate de 0% a
# 100% é igualmente plausível". Prior fraco/não-informativo, aplicado
# igualmente a TODOS os arquétipos (cada um roda sua própria posterior a
# partir dele - não há mistura de dados entre arquétipos).

alpha0, beta0 = 1, 1

# N_SIM = quantos números aleatórios sorteamos de CADA posterior pra estimar
# probabilidades por simulação Monte Carlo (não existe fórmula fechada
# simples pra comparar 3+ Betas de uma vez). Erro dessa aproximação ≈
# sqrt(p(1-p)/N_SIM); com 200_000 o erro máximo é ~0.11 p.p. (preciso o
# suficiente, sem deixar o script lento).
N_SIM = 200_000


def nome_arquivo_seguro(nome):
    """Converte o nome do arquétipo num nome de arquivo válido (sem espaços/acentos problemáticos)."""
    return re.sub(r"[^\w\-]+", "_", nome.strip())


def analisar_arquetipo(nome_arquetipo, variantes_lista):
    """Roda a análise bayesiana completa para UM arquétipo e retorna um resumo."""

    variantes = {f"Deck {i}": dict(d) for i, d in enumerate(variantes_lista, start=1)}
    for nome, d in variantes.items():
        d["vitorias"] = round(d["winrate"] * d["jogos"])
        d["derrotas"] = d["jogos"] - d["vitorias"]

    # --- posterior + amostragem Monte Carlo ---
    posteriors = {}
    amostras = {}
    for nome, d in variantes.items():
        a_post = alpha0 + d["vitorias"]
        b_post = beta0 + d["derrotas"]
        posteriors[nome] = (a_post, b_post)
        amostras[nome] = np.random.beta(a_post, b_post, size=N_SIM)

    nomes = list(variantes.keys())
    matriz_amostras = np.vstack([amostras[n] for n in nomes])
    indice_melhor = np.argmax(matriz_amostras, axis=0)
    melhor_em_cada_sim = np.max(matriz_amostras, axis=0)

    print("\n" + "#" * 90)
    print(f"ARQUÉTIPO: {nome_arquetipo}")
    print("#" * 90)

    # --- legenda de deck codes ---
    print("\nLEGENDA DE DECKS")
    print("-" * 90)
    for nome, d in variantes.items():
        print(f"{nome}:\n  {d['deck_code']}\n")

    # --- resumo por variante ---
    print("RESUMO POR VARIANTE")
    print("-" * 90)
    print(f"{'Variante':10s} {'Jogos':>7s} {'Winrate bruto':>14s} {'Média posterior':>17s} {'IC 95% (credível)':>20s}")
    for nome, (a, b) in posteriors.items():
        media_post = a / (a + b)
        ic_low, ic_high = stats.beta.ppf([0.025, 0.975], a, b)
        print(f"{nome:10s} {variantes[nome]['jogos']:7d} "
              f"{variantes[nome]['winrate']*100:13.1f}% "
              f"{media_post*100:16.1f}% "
              f"[{ic_low*100:5.1f}%, {ic_high*100:5.1f}%]")

    # --- P(é a melhor) ---
    print("\nP(variante é a MELHOR do arquétipo | dados)")
    print("-" * 90)
    probs_melhor = {}
    for i, nome in enumerate(nomes):
        prob_melhor = np.mean(indice_melhor == i)
        probs_melhor[nome] = prob_melhor
        print(f"  {nome:10s}: {prob_melhor*100:5.1f}%")

    # --- grupo estatisticamente competitivo ---
    # Quando o "vencedor" do ranking tem confiança baixa, normalmente não é
    # porque ele é claramente ruim, e sim porque VÁRIOS decks (em geral, os
    # de poucos jogos) estão genuinamente indistinguíveis entre si com o
    # volume de dados disponível. Em vez de só apontar 1 vencedor isolado,
    # listamos todos os decks cuja chance de serem o verdadeiro melhor
    # passa de um limiar mínimo - ou seja, que não podem ser descartados
    # como "claramente inferiores" aos demais.
    LIMIAR_COMPETITIVO = 0.05  # 5%: ajuste pra ser mais ou menos rigoroso
    grupo_competitivo = sorted(
        (n for n in nomes if probs_melhor[n] >= LIMIAR_COMPETITIVO),
        key=lambda n: -probs_melhor[n],
    )
    print(f"\nGRUPO ESTATISTICAMENTE COMPETITIVO (P(melhor) >= {LIMIAR_COMPETITIVO*100:.0f}%):")
    if len(grupo_competitivo) == 1:
        print(f"  Apenas {grupo_competitivo[0]} - vitória bem definida, sem concorrentes plausíveis.")
    else:
        for n in grupo_competitivo:
            print(f"  {n:10s}: P(melhor) = {probs_melhor[n]*100:5.1f}%  |  jogos = {variantes[n]['jogos']}")
        print(f"  -> Estes {len(grupo_competitivo)} decks não podem ser distinguidos com confiança entre si;")
        print(f"     o restante (jogos insuficientes ou winrate consistentemente mais baixo) pode ser descartado.")

    # --- perda esperada ---
    print("\nPERDA ESPERADA ao escolher cada variante (pontos percentuais de winrate)")
    print("-" * 90)
    for i, nome in enumerate(nomes):
        perda = np.mean(melhor_em_cada_sim - matriz_amostras[i])
        print(f"  {nome:10s}: {perda*100:5.2f} p.p.")

    # --- comparações par a par ---
    print("\nCOMPARAÇÕES PAR A PAR: P(linha > coluna)")
    print("-" * 90)
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            p = np.mean(amostras[nomes[i]] > amostras[nomes[j]])
            print(f"  P({nomes[i]} > {nomes[j]}) = {p*100:.1f}%")

    # --- melhor variante e seu deck code ---
    melhor_nome = max(probs_melhor, key=probs_melhor.get)
    melhor_deck_code = variantes[melhor_nome]["deck_code"]
    prob_vencedor = probs_melhor[melhor_nome]

    # Rótulo de confiança: o "vencedor" do ranking pode ter ganho por uma
    # margem ínfima (ex: 50.7% vs 49.3% - estatisticamente um empate) ou
    # por uma margem clara (ex: 99% - confiança real). O P(melhor) sozinho
    # já informa isso, mas o rótulo evita ter que ler os números toda vez.
    # Os limiares (90%/65%) são uma convenção arbitrária, ajuste se quiser
    # ser mais ou menos rigoroso.
    if prob_vencedor >= 0.90:
        confianca = "ALTA - diferença bem sustentada pelos dados"
    elif prob_vencedor >= 0.65:
        confianca = "MODERADA - vencedor provável, mas com incerteza real"
    else:
        confianca = "BAIXA - praticamente empate estatístico, não decida só por isso"

    print(f"\n>>> MELHOR VARIANTE: {melhor_nome}  (P(melhor) = {prob_vencedor*100:.1f}%)")
    print(f">>> CONFIANÇA: {confianca}")
    print(f">>> DECK CODE: {melhor_deck_code}")

    # --- gráfico das posteriors, salvo com o nome do arquétipo ---
    # O eixo X se ajusta à faixa onde as posteriors realmente vivem, em vez
    # de fixar [0,1] sempre. Isso importa porque com poucos jogos as
    # posteriors são largas (precisam do range quase completo), mas com
    # muitos jogos (ex: n > 10.000) elas ficam extremamente estreitas - usar
    # [0,1] faria todos os picos se amontoarem visualmente, escondendo as
    # diferenças reais entre os decks.
    limites_inf = [stats.beta.ppf(0.001, a, b) for a, b in posteriors.values()]
    limites_sup = [stats.beta.ppf(0.999, a, b) for a, b in posteriors.values()]
    faixa = max(limites_sup) - min(limites_inf)
    margem = 0.25 * faixa if faixa > 0 else 0.05  # 25% de respiro nas bordas
    x_min = max(0.0, min(limites_inf) - margem)
    x_max = min(1.0, max(limites_sup) + margem)

    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(9, 5))
    # P(melhor) entra na legenda de propósito: a ALTURA do pico reflete só
    # a precisão da estimativa (decks com mais jogos têm picos mais altos
    # e estreitos), não quem tem o winrate mais alto. Sem esse número ao
    # lado, é fácil olhar pro gráfico e achar que o pico mais alto "venceu"
    # - quando na real ele pode estar confiante sobre um valor MENOR.
    for nome, (a, b) in posteriors.items():
        rotulo = f"{nome} (n={variantes[nome]['jogos']}, P(melhor)={probs_melhor[nome]*100:.0f}%)"
        plt.plot(x, stats.beta.pdf(x, a, b), label=rotulo, linewidth=2)
    plt.xlabel("Winrate verdadeiro (p)")
    plt.ylabel("Densidade de probabilidade")
    plt.title(f"Distribuições posteriores do winrate - {nome_arquetipo}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    nome_arquivo_grafico = f"posteriors_{nome_arquivo_seguro(nome_arquetipo)}.png"
    plt.savefig(nome_arquivo_grafico, dpi=150)
    plt.close()
    print(f">>> Gráfico salvo em: {nome_arquivo_grafico}")

    return {
        "arquetipo": nome_arquetipo,
        "melhor_variante": melhor_nome,
        "melhor_deck_code": melhor_deck_code,
        "prob_melhor": prob_vencedor,
        "confianca": confianca,
        "grafico": nome_arquivo_grafico,
    }


# ---------------------------------------------------------------------------
# 3) RODAR PARA TODOS OS ARQUÉTIPOS
# ---------------------------------------------------------------------------
resultados = []
for nome_arquetipo, variantes_lista in arquetipos.items():
    resultados.append(analisar_arquetipo(nome_arquetipo, variantes_lista))

# ---------------------------------------------------------------------------
# 4) RESUMO FINAL - MELHOR DECK DE CADA ARQUÉTIPO (também salvo em .txt)
# ---------------------------------------------------------------------------
# Construímos as linhas do resumo numa lista primeiro, e só então printamos
# E gravamos em arquivo - assim o texto impresso na tela é IDÊNTICO ao
# salvo, sem duplicar a lógica de formatação em dois lugares.

linhas_resumo = []
linhas_resumo.append("=" * 90)
linhas_resumo.append("RESUMO FINAL - MELHOR VARIANTE DE CADA ARQUÉTIPO")
linhas_resumo.append("=" * 90)
for r in resultados:
    linhas_resumo.append(f"\n{r['arquetipo']}:")
    linhas_resumo.append(f"  Melhor variante : {r['melhor_variante']}  (P(melhor) = {r['prob_melhor']*100:.1f}%)")
    linhas_resumo.append(f"  Confiança       : {r['confianca']}")
    linhas_resumo.append(f"  Deck code       : {r['melhor_deck_code']}")
    linhas_resumo.append(f"  Gráfico salvo em: {r['grafico']}")

texto_resumo = "\n".join(linhas_resumo)

print("\n" + texto_resumo)

NOME_ARQUIVO_RESUMO = "resumo_final.txt"
with open(NOME_ARQUIVO_RESUMO, "w", encoding="utf-8") as f:
    f.write(texto_resumo + "\n")

print(f"\nResumo final também salvo em: {NOME_ARQUIVO_RESUMO}")