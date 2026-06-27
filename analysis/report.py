"""HTML report generation for the final analysis results."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    frame = df if max_rows is None else df.head(max_rows)
    return frame.to_html(index=False, border=0, classes="dataframe")


def generate_html_report(
    output_path: Path,
    deck_ranking: pd.DataFrame,
    archetype_ranking: pd.DataFrame,
    deck_stats: pd.DataFrame,
    archetype_stats: pd.DataFrame,
    archetype_top_decks: pd.DataFrame,
    graph_paths: dict[str, str],
    prior_alpha: float,
    prior_beta: float,
) -> Path:
    """Render a self-contained HTML report with tables, plots, and interpretation."""

    top10_archetypes = archetype_ranking.head(10)
    top20_decks = deck_ranking.head(20)
    top_decks_by_archetype = archetype_top_decks.copy()

    images_html = "".join(
        f'<figure><img src="{escape(path)}" alt="{escape(title)}"><figcaption>{escape(title)}</figcaption></figure>'
        for title, path in graph_paths.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hearthstone Bayesian Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; color: #1f2937; }}
    h1, h2, h3 {{ color: #111827; }}
    .callout {{ padding: 16px; border-left: 4px solid #2563eb; background: #eff6ff; margin: 16px 0; }}
    figure {{ margin: 20px 0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
    table.dataframe {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; font-size: 0.92rem; }}
    table.dataframe th, table.dataframe td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    table.dataframe th {{ background: #f3f4f6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
  <h1>Relatório Bayesiano de Hearthstone</h1>
  <div class="callout">
    <strong>Prior estimado:</strong> Beta({prior_alpha:.4f}, {prior_beta:.4f})<br>
    O modelo usa Beta-Binomial com Empirical Bayes, Monte Carlo vetorizado e score de confiabilidade que combina força posterior, variância, volume de jogos e popularidade do arquétipo.
  </div>

  <h2>Resumo Executivo</h2>
  <p>O deck mais confiável de cada arquétipo é selecionado pelo Reliability Score. O ranking global dos arquétipos usa a média ponderada dos melhores decks, ajustada pela popularidade total do arquétipo.</p>

  <div class="grid">
    <section>
      <h3>Top 10 Arquétipos</h3>
      {_table(top10_archetypes)}
    </section>
    <section>
      <h3>Top 20 Decks</h3>
      {_table(top20_decks)}
    </section>
  </div>

  <h2>Estatísticas</h2>
  <h3>Decks</h3>
  {_table(deck_stats)}
  <h3>Arquétipos</h3>
  {_table(archetype_stats)}

  <h2>Melhores decks por arquétipo</h2>
  <p>Esta tabela mostra, para cada arquétipo, os decks mais confiáveis pelas métricas bayesianas. Os rótulos abaixo são ordenados internamente por <em>Reliability Score</em> e já destacam o ranking dentro do arquétipo.</p>
  {_table(top_decks_by_archetype)}

  <h2>Visualizações</h2>
  {images_html}

  <h2>Interpretação</h2>
  <p>Decks com poucas partidas sofrem shrinkage em direção ao prior, enquanto decks com alta amostra preservam melhor o winrate observado. O probabilidade de melhor deck é estimada via simulação Monte Carlo com amostragem da posterior Beta de cada deck dentro do seu arquétipo.</p>
  <p>O score final penaliza baixa precisão estatística e baixa exposição amostral, evitando que um winrate alto com pouco volume supere decks mais estáveis e consistentes.</p>

  <h2>Conclusões</h2>
  <p>Use o ranking de decks para escolher a lista mais forte dentro do arquétipo e o ranking de arquétipos para comparar qual pacote estatístico parece mais robusto no meta atual.</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
