"""Entry point for the full Hearthstone Bayesian analysis pipeline."""

from __future__ import annotations

from os.path import relpath
from pathlib import Path

import pandas as pd

from analysis.loader import prepare_dataset
from analysis.bayesian import (
    add_group_threshold_probabilities,
    enrich_with_posterior_metrics,
    estimate_beta_prior,
)
from analysis.montecarlo import probability_of_best
from analysis.ranking import build_archetype_ranking, build_archetype_top_decks, build_deck_ranking
from analysis.reliability import compute_reliability_score
from analysis.report import generate_html_report
from analysis.statistics import compute_archetype_statistics
from config import AnalysisConfig
from visualization.forest import plot_forest
from visualization.heatmap import plot_heatmap
from visualization.posterior import plot_posterior_curves, plot_shrinkage
from visualization.probability_best import plot_probability_best
from visualization.ranking import plot_archetype_rankings
from visualization.scatter import plot_scatter
from utils.io import save_dataframe_csv
from utils.logger import get_logger
from utils.helpers import configure_utf8_console
from utils.paths import ensure_project_directories


def run_analysis(config: AnalysisConfig | None = None) -> dict[str, Path]:
    """Run the full analysis pipeline and export every deliverable."""

    config = config or AnalysisConfig()
    configure_utf8_console()
    ensure_project_directories()
    logger = get_logger("hearthstone-analysis", config.paths.output_dir / "analysis.log")

    logger.info("Carregando dados de entrada")
    deck_frame = prepare_dataset()

    logger.info("Estimando prior Empirical Bayes")
    prior = estimate_beta_prior(deck_frame)
    logger.info("Prior estimado: alpha=%.4f | beta=%.4f | metodo=%s", prior.alpha, prior.beta, prior.method)

    logger.info("Calculando métricas posteriores por deck")
    deck_frame = enrich_with_posterior_metrics(deck_frame, prior.alpha, prior.beta, config.thresholds)
    deck_frame = add_group_threshold_probabilities(deck_frame)

    logger.info("Executando Monte Carlo com %d amostras", config.mc_samples)
    deck_frame = probability_of_best(deck_frame, config.mc_samples, config.seed)
    deck_frame = compute_reliability_score(deck_frame, config.games_k)

    logger.info("Calculando estatísticas por arquétipo")
    archetype_stats = compute_archetype_statistics(deck_frame)
    archetype_ranking = build_archetype_ranking(deck_frame, archetype_stats)
    deck_ranking = build_deck_ranking(deck_frame)
    archetype_top_decks = build_archetype_top_decks(deck_ranking, top_n=3)

    logger.info("Gerando gráficos")
    graph_paths: dict[str, str] = {}
    figure_paths = {
        "scatter": plot_scatter(deck_ranking, config.paths.output_figures_dir),
        "forest": plot_forest(deck_ranking, config.paths.output_figures_dir, config.forest_plot_size),
        "posterior": plot_posterior_curves(deck_ranking, config.paths.output_figures_dir, config.top_posterior_curves),
        "probability_best": plot_probability_best(deck_ranking, config.paths.output_figures_dir, config.top_decks_global),
        "heatmap": plot_heatmap(archetype_ranking, config.paths.output_figures_dir),
        "shrinkage": plot_shrinkage(deck_ranking, config.paths.output_figures_dir),
    }
    graph_paths = {
        key: Path(relpath(path, start=config.paths.output_reports_dir)).as_posix()
        for key, path in figure_paths.items()
    }
    for path in plot_archetype_rankings(deck_ranking, config.paths.output_figures_dir / "archetypes", top_n=5):
        graph_paths[f"archetype_{path.stem}"] = Path(relpath(path, start=config.paths.output_reports_dir)).as_posix()

    logger.info("Exportando CSVs")
    save_dataframe_csv(deck_frame, config.paths.output_csv_dir / "analysis.csv")
    save_dataframe_csv(deck_ranking, config.paths.output_csv_dir / "deck_statistics.csv")
    save_dataframe_csv(archetype_ranking, config.paths.output_csv_dir / "archetype_statistics.csv")
    save_dataframe_csv(archetype_top_decks, config.paths.output_csv_dir / "archetype_top_decks.csv")
    save_dataframe_csv(
        pd.concat([deck_ranking.assign(kind="deck"), archetype_ranking.assign(kind="archetype")], ignore_index=True, sort=False),
        config.paths.output_csv_dir / "ranking.csv",
    )

    logger.info("Gerando relatório HTML")
    report_path = generate_html_report(
        config.paths.output_reports_dir / "report.html",
        deck_ranking,
        archetype_ranking,
        deck_ranking,
        archetype_ranking,
        archetype_top_decks,
        graph_paths,
        prior.alpha,
        prior.beta,
    )

    summary_lines = ["Resumo final por arquétipo", ""]
    for archetype, group in deck_ranking.groupby("archetype", sort=False):
        top = group.sort_values(["reliability_score", "posterior_mean", "jogos"], ascending=[False, False, False]).head(3)
        summary_lines.append(archetype)
        for row in top.itertuples():
            summary_lines.append(
                f"#{int(row.deck_rank_in_archetype)} | score={row.reliability_score:.3f} | wr={row.winrate_observed:.3f} | "
                f"jogos={int(row.jogos)} | post={row.posterior_mean:.3f} | IC95%=[{row.credible_95_low:.3f}, {row.credible_95_high:.3f}] | "
                f"P(best deck)={row.prob_best*100:.1f}%"
            )
        summary_lines.append("")

    summary_path = config.paths.output_reports_dir / "resumo_final.txt"
    summary_text = "\n".join(summary_lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    Path("resumo_final.txt").write_text(summary_text, encoding="utf-8")

    logger.info("Análise concluída com sucesso")
    logger.info("Melhor arquétipo: %s", archetype_ranking.iloc[0]["archetype"])
    logger.info("Melhor deck: %s", deck_ranking.iloc[0]["deck_code"])

    return {
        "analysis_csv": config.paths.output_csv_dir / "analysis.csv",
        "deck_statistics_csv": config.paths.output_csv_dir / "deck_statistics.csv",
        "archetype_statistics_csv": config.paths.output_csv_dir / "archetype_statistics.csv",
        "ranking_csv": config.paths.output_csv_dir / "ranking.csv",
        "report_html": report_path,
        "summary_txt": summary_path,
        "graphs_dir": config.paths.output_figures_dir,
        "log_file": config.paths.output_dir / "analysis.log",
    }


def main() -> None:
    """Console entry point."""

    run_analysis()


if __name__ == "__main__":
    main()
