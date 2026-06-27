# Hearthstone Bayesian Analysis

Projeto em Python para coletar, organizar e analisar decks de Hearthstone com um modelo Bayesiano Beta-Binomial. O foco é identificar, para cada arquétipo, o deck mais confiável considerando winrate, volume de jogos e popularidade do arquétipo.

## Visão Geral

O projeto está separado em três etapas:

1. `scraping` coleta os dados do HSGuru e salva os JSONs brutos em `data/raw/`.
2. `analysis` carrega os JSONs e calcula posterior Bayesiana, ranking, estatísticas e confiabilidade.
3. `visualization` gera os gráficos e o relatório final em `data/output/`.

## Estrutura

```text
HearthstoneAnalysis/
├── main.py
├── config.py
├── scraping/
├── analysis/
├── visualization/
├── utils/
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
└── tests/
```

## Como Rodar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Coletar arquétipos

```bash
python Scrape-Archetypes.py
```

Isso cria ou atualiza `data/raw/archetypes.json`.

### 3. Coletar decks

```bash
python Scrape-Decks.py
```

Isso cria ou atualiza `data/raw/decks.json`.

### 4. Executar a análise

```bash
python main.py
```

Esse comando carrega os JSONs, estima o prior, executa Monte Carlo, calcula as métricas, gera os rankings, produz os gráficos e exporta os relatórios.

## Saídas Geradas

Os resultados são salvos em `data/output/`:

- `data/output/csv/analysis.csv`
- `data/output/csv/ranking.csv`
- `data/output/csv/deck_statistics.csv`
- `data/output/csv/archetype_statistics.csv`
- `data/output/csv/archetype_top_decks.csv`
- `data/output/figures/*.png`
- `data/output/reports/report.html`
- `data/output/reports/resumo_final.txt`

Também é mantida uma cópia de compatibilidade de `resumo_final.txt` na raiz do projeto.

## Scripts Legados

Os arquivos da raiz continuam funcionando como wrappers de compatibilidade:

- `Scrape-Archetypes.py`
- `Scrape-Decks.py`
- `Bayesian-Analysis.py`
- `Formatting-Decks.py`

## Parâmetros Configuráveis

Todos os parâmetros principais ficam centralizados em [config.py](config.py).

### `AnalysisConfig`

- `seed`: seed do gerador aleatório para reproduzibilidade.
- `mc_samples`: quantidade de amostras Monte Carlo por deck.
- `games_k`: constante de suavização usada no `Reliability Score`.
- `thresholds`: limites usados para calcular probabilidades como `P(winrate > 50%)`.
- `min_games`: mínimo de jogos usado no scraping.
- `prior_eps`: epsilon numérico para estabilidade.
- `headers`: headers HTTP usados no scraping.
- `top_decks_global`: quantidade de decks exibidos nos gráficos globais.
- `top_archetypes_global`: quantidade de arquétipos exibidos nos rankings globais.
- `top_posterior_curves`: quantidade de curvas posteriores exibidas nos gráficos.
- `forest_plot_size`: quantidade de decks mostrados no forest plot.

### Paths centralizados

O arquivo `config.py` também expõe os caminhos do projeto:

- `data/raw/` para JSONs brutos.
- `data/processed/` para artefatos intermediários.
- `data/output/csv/` para tabelas finais.
- `data/output/figures/` para imagens.
- `data/output/reports/` para HTML e resumos.

## Testes

Executar os testes básicos:

```bash
python -m unittest discover -s tests
```

## Observações

- O projeto usa `pathlib.Path` para portabilidade entre Windows, Linux e macOS.
- A saída do terminal é configurada para UTF-8 nas entradas principais.
- A análise é modular e preparada para expansões futuras, como novos rankings, novas métricas ou outras fontes de dados.

