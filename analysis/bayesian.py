"""Bayesian model fitting and posterior metric calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import betaln

from config import (
    METHOD_OF_MOMENTS_VARIANCE_FLOOR,
    PRIOR_BOUNDS,
    PRIOR_METHOD,
    PRIOR_START_FLOOR,
)


@dataclass(slots=True)
class PriorEstimate:
    """Empirical Beta prior estimated from the full dataset."""

    alpha: float
    beta: float
    method: str
    objective: float


def _safe_moments_prior(wins: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Build a stable initial prior guess from method of moments."""

    games = wins + losses
    if np.sum(games) <= 0:
        return PRIOR_START_FLOOR, PRIOR_START_FLOOR

    observed_mean = np.sum(wins) / np.sum(games)
    proportions = np.divide(wins, games, out=np.zeros_like(wins, dtype=float), where=games > 0)
    weighted_var = np.average((proportions - observed_mean) ** 2, weights=np.maximum(games, 1.0))
    weighted_var = float(max(weighted_var, METHOD_OF_MOMENTS_VARIANCE_FLOOR))

    concentration = observed_mean * (1.0 - observed_mean) / weighted_var - 1.0
    concentration = float(max(concentration, PRIOR_START_FLOOR))

    alpha = max(observed_mean * concentration, PRIOR_START_FLOOR)
    beta = max((1.0 - observed_mean) * concentration, PRIOR_START_FLOOR)
    return alpha, beta


def estimate_beta_prior(deck_frame: pd.DataFrame) -> PriorEstimate:
    """Estimate the Beta prior with empirical Bayes over all observed decks.

    The optimizer maximizes the Beta-Binomial marginal likelihood across the
    complete deck sample. A method-of-moments prior is used as the starting point
    and as a fallback if the optimizer fails.
    """

    wins = deck_frame["wins"].to_numpy(dtype=float)
    losses = deck_frame["losses"].to_numpy(dtype=float)

    init_alpha, init_beta = _safe_moments_prior(wins, losses)

    def objective(log_params: np.ndarray) -> float:
        alpha = float(np.exp(log_params[0]))
        beta = float(np.exp(log_params[1]))
        return -float(np.sum(betaln(wins + alpha, losses + beta) - betaln(alpha, beta)))

    result = optimize.minimize(
        objective,
        x0=np.log([init_alpha, init_beta]),
        method="L-BFGS-B",
        bounds=[PRIOR_BOUNDS, PRIOR_BOUNDS],
    )

    if result.success:
        alpha, beta = np.exp(result.x)
        return PriorEstimate(float(alpha), float(beta), PRIOR_METHOD, float(result.fun))

    fallback_objective = objective(np.log([init_alpha, init_beta]))
    return PriorEstimate(float(init_alpha), float(init_beta), "method_of_moments", float(fallback_objective))


def enrich_with_posterior_metrics(
    deck_frame: pd.DataFrame,
    alpha0: float,
    beta0: float,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Attach deck-level Bayesian posterior metrics to the input frame."""

    frame = deck_frame.copy()

    alpha = alpha0 + frame["wins"].to_numpy(dtype=float)
    beta = beta0 + frame["losses"].to_numpy(dtype=float)

    total = alpha + beta
    posterior_mean = alpha / total
    posterior_var = (alpha * beta) / ((total**2) * (total + 1.0))
    posterior_std = np.sqrt(posterior_var)
    posterior_median = stats.beta.ppf(0.5, alpha, beta)

    frame["posterior_alpha"] = alpha
    frame["posterior_beta"] = beta
    frame["posterior_mean"] = posterior_mean
    frame["posterior_variance"] = posterior_var
    frame["posterior_std"] = posterior_std
    frame["posterior_median"] = posterior_median
    frame["credible_95_low"] = stats.beta.ppf(0.025, alpha, beta)
    frame["credible_95_high"] = stats.beta.ppf(0.975, alpha, beta)
    frame["credible_99_low"] = stats.beta.ppf(0.005, alpha, beta)
    frame["credible_99_high"] = stats.beta.ppf(0.995, alpha, beta)
    frame["expected_wins"] = posterior_mean * frame["jogos"].to_numpy(dtype=float)
    frame["expected_losses"] = frame["jogos"].to_numpy(dtype=float) - frame["expected_wins"]
    frame["shrinkage"] = frame["posterior_mean"] - frame["winrate_observed"]

    for threshold in thresholds:
        probability_column = f"prob_gt_{int(round(threshold * 100))}"
        frame[probability_column] = 1.0 - stats.beta.cdf(threshold, alpha, beta)

    return frame


def add_group_threshold_probabilities(deck_frame: pd.DataFrame) -> pd.DataFrame:
    """Add archetype-relative probabilities to each deck row."""

    frame = deck_frame.copy()
    archetype_mean = frame.groupby("archetype")["winrate_observed"].transform("mean").to_numpy(dtype=float)
    best_observed = frame.groupby("archetype")["winrate_observed"].transform("max").to_numpy(dtype=float)

    frame["prob_gt_archetype_mean"] = 1.0 - stats.beta.cdf(
        archetype_mean,
        frame["posterior_alpha"].to_numpy(dtype=float),
        frame["posterior_beta"].to_numpy(dtype=float),
    )
    frame["prob_gt_best_observed"] = 1.0 - stats.beta.cdf(
        best_observed,
        frame["posterior_alpha"].to_numpy(dtype=float),
        frame["posterior_beta"].to_numpy(dtype=float),
    )

    return frame
