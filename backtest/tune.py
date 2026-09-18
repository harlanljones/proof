"""Shrinkage-constant tuning with a held-out season — no test-set peeking.

The v1 constants were borrowed from Carleton's stabilization work, which
used league-mean priors; a player-specific prior has lower variance, so the
optimal k is plausibly larger (audit §15). Rather than hand-tuning on the
backtest we report — which would overfit it — the global shrinkage
multiplier is chosen on 2023–2024 and *validated* on 2025, which the
tuning loop never touches.

One coarse multiplier for all events, not seven free constants: with three
seasons of data, a 1-D grid is what the evidence can honestly support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.walkforward import run_backtest
from backtest.walkforward_pitchers import run_backtest_pitchers_fip

TRAIN_SEASONS = [2023, 2024]
HOLDOUT_SEASONS = [2025]
GRID = [0.5, 1.0, 2.0, 4.0]


def weighted_rmse(
    results: pd.DataFrame,
    system: str = "proof",
    target: str = "woba",
    weight: str = "PA_ros",
) -> float:
    err = results[f"pred_{target}_{system}"] - results[f"{target}_actual_ros"]
    return float(np.sqrt(np.average(err**2, weights=results[weight])))


def tune() -> tuple[pd.DataFrame, float]:
    rows = []
    for k_mult in GRID:
        res = run_backtest(TRAIN_SEASONS, k_mult=k_mult)
        rows.append(
            {
                "k_mult": k_mult,
                "rmse_pa_weighted_train": weighted_rmse(res),
                "rmse_prior_baseline_train": weighted_rmse(res, "prior"),
            }
        )
        print(
            f"  k_mult={k_mult}: train wRMSE={rows[-1]['rmse_pa_weighted_train']:.5f}"
        )
    table = pd.DataFrame(rows)
    best = float(table.loc[table["rmse_pa_weighted_train"].idxmin(), "k_mult"])
    return table, best


def tune_pitchers() -> tuple[pd.DataFrame, float]:
    """Same protocol as hitters: grid on 2023-24, holdout 2025 untouched."""
    rows = []
    for k_mult in GRID:
        res = run_backtest_pitchers_fip(TRAIN_SEASONS, k_mult=k_mult)
        rows.append(
            {
                "k_mult": k_mult,
                "rmse_bf_weighted_train": weighted_rmse(res, "proof", "fip", "BF_ros"),
                "rmse_prior_baseline_train": weighted_rmse(
                    res, "prior", "fip", "BF_ros"
                ),
                "rmse_eb_league_train": weighted_rmse(
                    res, "eb_league", "fip", "BF_ros"
                ),
            }
        )
        print(
            f"  k_mult={k_mult}: train wRMSE={rows[-1]['rmse_bf_weighted_train']:.5f}"
        )
    table = pd.DataFrame(rows)
    best = float(table.loc[table["rmse_bf_weighted_train"].idxmin(), "k_mult"])
    return table, best
