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
    "End of Turnadin": [
        {"deck_code": "AAECAZ8FBPD+BsODB8avB+XBBw3JoAS6lgfXlwfOmwf1rwfiwQfqwQf2wQeDwgfmxQerxge66AfQ6gcAAA==", "winrate": 0.626, "jogos": 3408},
        {"deck_code": "AAECAZ8FBvD+BsODB8avB+XBB6vGB7roBwzJoATTnga6lgfOmwf1rwfiwQfowQfqwQf2wQeDwgfmxQfQ6gcAAA==", "winrate": 0.621, "jogos": 27611},
        {"deck_code": "AAECAZ8FCPD+BsODB+6oB++oB/CoB/WvB+XBB6vGBwvJoAS6lgfLqQfErge+sgfiwQfowQfqwQf2wQeDwgfmxQcAAA==", "winrate": 0.610, "jogos": 2715},
        {"deck_code": "AAECAZ8FBvD+BsODB+6oB++oB/CoB+XBBwzJoATTnga6lgfErgf1rwe+sgfiwQfowQfqwQf2wQeDwgfmxQcAAA==", "winrate": 0.601, "jogos": 809},
        {"deck_code": "AAECAZ8FCs6eBvD+BsODB+6oB++oB/CoB/WvB42+B+XBB6vGBwrJoAS6lgfLqQfErge+sgfiwQfowQfqwQf2wQeDwgcAAA==", "winrate": 0.597, "jogos": 1186},
        {"deck_code": "AAECAZ8FBvD+BsODB+6oB++oB/CoB+XBBwzJoAS6lgfLqQfErgf1rwe+sgfiwQfowQfqwQf2wQeDwgerxgcAAA==", "winrate": 0.580, "jogos": 4398},
    ],
    "No Hand Hunter": [
        {"deck_code": "AAECAR8EmacHmqcHm6cHxbEHDamfBKqfBNOeBq+SB4WVB86bB+6fB5CnB5inB8u2B7TAB7nAB7vABwAA", "winrate": 0.565, "jogos": 6780},
        {"deck_code": "AAECAR8EmacHmqcHm6cHxbEHDamfBKqfBKj9Bq+SB4WVB86bB+6fB5CnB5inB8u2B7TAB7nAB7vABwAA", "winrate": 0.543, "jogos": 11572},
        {"deck_code": "AAECAR8EmacHmqcHm6cHxbEHDamfBKqfBK+SB4WVB9eXB86bB+6fB5CnB5inB7TAB7nAB7vAB97EBwAA", "winrate": 0.532, "jogos": 3306},
    ],
    "Token Druid": [
        {"deck_code": "AAECAZICBMODB6+HB7ifB+DABw2unwSB1ASIgweuhweSlweUlwfXlwfanQeqrwfXwAfbwAfswAf2wQcAAA==", "winrate": 0.558, "jogos": 1727},
        {"deck_code": "AAECAZICCJKDB8ODB6+HB4KYB+DAB93EB+XEB/bJBwuunwSB1ASIgweuhweSlweUlwfXlweqrwe+sgfXwAf2wQcAAA==", "winrate": 0.542, "jogos": 1551},
    ],
    "Quest Mage": [
        {"deck_code": "AAECAf0EBu/8BsODB7OHB6ebB/KyB5PaBwyb8gbxkQewmwf6mwfVnQfRpgfLtgf5wweGxAeSxAeG4AecgggAAA==", "winrate": 0.538, "jogos": 5066},
        {"deck_code": "AAECAf0EBu/8BsODB6ebB/mbB/KyB5PaBwyb8gbxkQewmwf6mwfVnQfRpgfLtgf5wweGxAeSxAeG4AecgggAAA==", "winrate": 0.534, "jogos": 8711},
        {"deck_code": "AAECAf0EBu/8BsODB7OHB6ebB/WlB/KyBwyb8gbxkQewmwf6mwfVnQfRpgfLtgf5wweGxAeSxAeG4AecgggAAA==", "winrate": 0.530, "jogos": 2752},
        {"deck_code": "AAECAf0EBu/8BsODB6ebB/WlB+yqB/KyBwyb8ga1+gbxkQewmwf6mwfRpgfLtgf5wweGxAeSxAeG4AecgggAAA==", "winrate": 0.492, "jogos": 1676},
    ],
    "Leyline Mage": [
        {"deck_code": "AAECAf0EBJegBIi+B/rEB//fBw2zhwfopQeLsQfWvAeSxAfnxAfoxAfwxAfzxAf0xAf7xAesxgeT2gcAAA==", "winrate": 0.534, "jogos": 8702},
        {"deck_code": "AAECAf0EBJegBMODB/rEB6rJBw2zhwf1pQfWsgeGxAeSxAfnxAfoxAfwxAfzxAf0xAf7xAesxgeT2gcAAA==", "winrate": 0.532, "jogos": 10012},
    ],
    "Companion Hunter": [
        {"deck_code": "AAECAR8InaAEqYEHmacHmqcHm6cHxbEHg8AH5MQHC6mfBOKJB6+SB+6fB8u2B7vAB97EB+DEB+PEB/nEB6H9BwAA", "winrate": 0.535, "jogos": 4755},
        {"deck_code": "AAECAR8KnaAEzZ4GqYEHmacHmqcHm6cHxbEHg8AH5MQHof0HCqmfBOKJB6+SB+6fB8u2B7vAB97EB+DEB+PEB/nEBwAA", "winrate": 0.531, "jogos": 4822},
        {"deck_code": "AAECAR8KnaAEzp4GqYEHmacHmqcHm6cHxbEHg8AH5MQHof0HCqmfBOKJB6+SB+6fB8u2B7vAB97EB+DEB+PEB/nEBwAA", "winrate": 0.526, "jogos": 3333},
        {"deck_code": "AAECAR8KnaAEqYEHw4MHmacHmqcHm6cHxbEHg8AH5MQHof0HCqmfBOKJB6+SB+6fB8u2B7vAB97EB+DEB+PEB/nEBwAA", "winrate": 0.523, "jogos": 5213},
    ],
    "Dude Paladin": [
        {"deck_code": "AAECAZ8FCPD+BsODB9eXB42dB8avB+XBB4TEB+jFBwvJoASU9QW6lgfOmwf1rwf2wQeDwgfkxQfmxQfnxQfD4wcAAA==", "winrate": 0.625, "jogos": 5273},
        {"deck_code": "AAECAZ8FBMODB8avB+XBB+jFBw3JoASU9QW6lgfXlwfOmweNnQf1rwf2wQeDwgfkxQfmxQfnxQfD4wcAAA==", "winrate": 0.622, "jogos": 2430},
        {"deck_code": "AAECAZ8FBsODB42dB8avB+XBB4TEB+jFBwzJoASU9QW6lgfXlwfOmwf1rwf2wQeDwgfkxQfmxQfnxQfD4wcAAA==", "winrate": 0.611, "jogos": 31621},
    ],
    "Dragon Warrior": [
        {"deck_code": "AAECAQcCi6AE0LIHDuPmBqr8Bqv8BveDB+iHB9KXB7etB4+xB+yyB4S9B7XAB5XCB5vCB5zCBwAA", "winrate": 0.606, "jogos": 4044},
        {"deck_code": "AAECAQcCi6AE0LIHDuPmBqr8Bqv8BuiHB9KXB7etB4+xB+yyB4S9B7XAB5XCB5vCB5zCB/nDBwAA", "winrate": 0.592, "jogos": 22623},
        {"deck_code": "AAECAQcEi6AEyZ4G94MHj7EHDePmBqr8Bqv8BuiHB9KXB7etB+yyB4S9B7XAB5XCB5vCB5zCB/nDBwAA", "winrate": 0.587, "jogos": 4004},
        {"deck_code": "AAECAQcAD+PmBqr8Bqv8BveDB8eHB+iHB9KXB7etB9CyB+yyB4S9B7XAB5XCB5vCB5zCBwAA", "winrate": 0.583, "jogos": 5878},
    ],
    "Harold Shaman": [
        {"deck_code": "AAECAaoICq+fBP2fBMODB4KYB9umB9+mB+WmB9C/B4LUB5vUBwrmlgf1rAexsAe8sQePvgfDwAfJwAf3wAf2wQfm/QcAAA==", "winrate": 0.589, "jogos": 13212},
        {"deck_code": "AAECAaoICK+fBMODB9umB9+mB+WmB9C/B4LUB5vUBwv9nwTmlgf1rAexsAe8sQePvgfDwAfJwAf3wAf2wQfm/QcAAA==", "winrate": 0.575, "jogos": 11643},
        {"deck_code": "AAECAaoICq+fBMODB4KYB9umB9+mB+WmB7GwB9C/B4LUB5vUBwr9nwTmlgf1rAe8sQePvgfDwAfJwAf3wAf2wQfm/QcAAA==", "winrate": 0.562, "jogos": 5864},
    ],
    "No Minion DH": [
        {"deck_code": "AAECAea5Awa0lweKqgeSqgeTqgensQeUvwcM4fgF3v8G/oMHqocHtpcH550HnrEHobEHwLEH6LEHkr8Hlb8HAAA=", "winrate": 0.563, "jogos": 1180},
        {"deck_code": "AAECAea5Awa0lweKqgeSqgeTqgensQeUvwcM4fgF3v8G/oMHqocHtpcH550HnrEHobEHv7EHwLEHkr8Hlb8HAAA=", "winrate": 0.562, "jogos": 859},
    ],
    "Burn Mage": [
        {"deck_code": "AAECAf0EBJegBIi+B9fDB5HGBw39ngSF5gbHhweflgfopQeLsQfWsgfWvAeGxAeSxAebxAesxgeT2gcAAA==", "winrate": 0.573, "jogos": 2823},
        {"deck_code": "AAECAf0EBpegBLOHB4i+B9fDB5HGB5PaBwz9ngSF5gbHhweflgfopQeLsQfWsgfWvAeGxAeSxAebxAesxgcAAA==", "winrate": 0.570, "jogos": 6432},
        {"deck_code": "AAECAf0EBJegBMODB7OHB4i+Bw39ngSF5gayhwfHhweflgfOmwfopQeLsQeSxAebxAesxgeqyQeT2gcAAA==", "winrate": 0.553, "jogos": 3480},
        {"deck_code": "AAECAf0EApegBIi+Bw6yhwezhwfHhweflgfopQeLsQfWsgfWvAeGxAeSxAebxAesxgeqyQeT2gcAAA==", "winrate": 0.553, "jogos": 1970},
        {"deck_code": "AAECAf0EAv2eBIi+Bw6yhwezhwfHhweflgfopQeLsQfWsgfWvAeGxAeSxAebxAesxgeqyQeT2gcAAA==", "winrate": 0.501, "jogos": 1719},
    ],
    "Harold Rogue": [
        {"deck_code": "AAECAaIHCqGBB5KDB8ODB4KYB+ylB9C/B/bJB4rUB5vUB4jZBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.563, "jogos": 44508},
        {"deck_code": "AAECAaIHCqGBB5KDB4KYB9GdB+ylB9C/B/bJB4rUB5vUB4jZBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.555, "jogos": 57662},
        {"deck_code": "AAECAaIHCtmiBqGBB5KDB4KYB+ylB9C/B/bJB4rUB5vUB4jZBwqRnwT3nwTTngaQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.552, "jogos": 25005},
        {"deck_code": "AAECAaIHDNOeBqGBB5KDB8ODB4KYB9GdB+ylB9C/B/bJB4rUB5vUB4jZBwmRnwT3nwSQgwfBlwfHrge0wQfAwQedxQfVxQcAAA==", "winrate": 0.546, "jogos": 20456},
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
N_SIM = 1_000_000
 
 
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
    print(f"\n>>> MELHOR VARIANTE: {melhor_nome}  (P(melhor) = {probs_melhor[melhor_nome]*100:.1f}%)")
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
    for nome, (a, b) in posteriors.items():
        plt.plot(x, stats.beta.pdf(x, a, b), label=f"{nome} (n={variantes[nome]['jogos']})", linewidth=2)
    plt.xlabel("Winrate verdadeiro (p)")
    plt.ylabel("Densidade de probabilidade")
    plt.title(f"Distribuições posteriores do winrate - {nome_arquetipo}")
    plt.legend()
    plt.tight_layout()
    nome_arquivo_grafico = f"posteriors_{nome_arquivo_seguro(nome_arquetipo)}.png"
    plt.savefig(nome_arquivo_grafico, dpi=150)
    plt.close()
    print(f">>> Gráfico salvo em: {nome_arquivo_grafico}")
 
    return {
        "arquetipo": nome_arquetipo,
        "melhor_variante": melhor_nome,
        "melhor_deck_code": melhor_deck_code,
        "prob_melhor": probs_melhor[melhor_nome],
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
    linhas_resumo.append(f"  Deck code       : {r['melhor_deck_code']}")
    linhas_resumo.append(f"  Gráfico salvo em: {r['grafico']}")
 
texto_resumo = "\n".join(linhas_resumo)
 
print("\n" + texto_resumo)
 
NOME_ARQUIVO_RESUMO = "resumo_final.txt"
with open(NOME_ARQUIVO_RESUMO, "w", encoding="utf-8") as f:
    f.write(texto_resumo + "\n")
 
print(f"\nResumo final também salvo em: {NOME_ARQUIVO_RESUMO}")