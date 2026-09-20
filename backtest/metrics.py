"""Scoring: RMSE/MAE (playing-time-weighted and plain), calibration, plots.

Generic over target ("woba" for hitters, "era" for pitchers); the result
frame convention is pred_{target}_{system}, {target}_actual_ros, and a
playing-time weight column (PA_ros / BF_ros).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "writeup" / "results"


def _wrmse(err: pd.Series, w: pd.Series) -> float:
    return float(np.sqrt(np.average(err**2, weights=w)))


def error_table(
    results: pd.DataFrame, target: str, systems: list[str], weight: str
) -> pd.DataFrame:
    rows = []
    for label, grp in [("ALL", results)] + list(results.groupby("season")):
        for system in systems:
            err = grp[f"pred_{target}_{system}"] - grp[f"{target}_actual_ros"]
            rows.append(
                {
                    "season": label,
                    "system": system,
                    "n": len(grp),
                    "rmse": float(np.sqrt((err**2).mean())),
                    "rmse_weighted": _wrmse(err, grp[weight]),
                    "mae": float(err.abs().mean()),
                    "bias": float(err.mean()),
                }
            )
    df = pd.DataFrame(rows)
    df["season"] = df["season"].astype(str)
    return df.sort_values(["season", "rmse"]).reset_index(drop=True)


def _sd_col(target: str) -> str:
    """ERA's predictive interval carries an extra defense/sequencing variance
    term (models.pitchers.SIGMA_ERA_EXTRA2) — scoring ERA against the FIP sd
    was the source of its underconfident calibration."""
    return "sd_proof_era" if target == "era" else "sd_proof"


def calibration(
    results: pd.DataFrame, target: str, levels: tuple[float, ...] = (0.5, 0.8, 0.95)
) -> pd.DataFrame:
    rows = []
    sd = results[_sd_col(target)]
    for lv in levels:
        z = stats.norm.ppf((1 + lv) / 2)
        covered = (
            results[f"{target}_actual_ros"] - results[f"pred_{target}_proof"]
        ).abs() <= z * sd
        rows.append(
            {"nominal": lv, "empirical": float(covered.mean()), "n": len(results)}
        )
    return pd.DataFrame(rows)


def plot_rmse_by_checkpoint(
    results: pd.DataFrame, target: str, systems: list[str], weight: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for system in systems:
        g = results.copy()
        g["sq"] = (g[f"pred_{target}_{system}"] - g[f"{target}_actual_ros"]) ** 2
        rmse = g.groupby("checkpoint").apply(
            lambda x: np.sqrt(np.average(x["sq"], weights=x[weight])),
            include_groups=False,
        )
        ax.plot(rmse.index, rmse.values, marker="o", label=system)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel(f"weighted RMSE (ROS {target})")
    ax.set_title(f"Rest-of-season {target} projection error by checkpoint, 2023-2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_calibration(results: pd.DataFrame, target: str, out: Path) -> None:
    z = (results[f"{target}_actual_ros"] - results[f"pred_{target}_proof"]) / results[
        _sd_col(target)
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = np.linspace(-3.5, 3.5, 200)
    ax.hist(z, bins=40, density=True, alpha=0.6, label="backtest residuals")
    ax.plot(xs, stats.norm.pdf(xs), "k-", lw=2, label="N(0,1)")
    ax.set_xlabel("(actual - projected) / predictive sd")
    ax.set_title(f"PROOF calibration: standardized ROS {target} errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_vs_pt(
    results: pd.DataFrame, target: str, weight: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    err = (results[f"pred_{target}_proof"] - results[f"{target}_actual_ros"]).abs()
    ax.scatter(results[weight], err, s=6, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlabel(f"ROS {weight.replace('_ros', '')} (log scale)")
    ax.set_ylabel(f"|{target} error|")
    ax.set_title(f"PROOF error vs. remaining playing time ({target})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_all(
    results: pd.DataFrame,
    target: str,
    systems: list[str],
    weight: str,
    prefix: str,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_DIR / f"{prefix}_predictions.parquet", index=False)

    errors = error_table(results, target, systems, weight)
    errors.to_csv(RESULTS_DIR / f"{prefix}_metrics_by_season.csv", index=False)

    calib = calibration(results, target)
    calib.to_csv(RESULTS_DIR / f"{prefix}_calibration.csv", index=False)

    plot_rmse_by_checkpoint(
        results,
        target,
        systems,
        weight,
        RESULTS_DIR / f"{prefix}_rmse_by_checkpoint.png",
    )
    plot_calibration(results, target, RESULTS_DIR / f"{prefix}_calibration.png")
    plot_error_vs_pt(results, target, weight, RESULTS_DIR / f"{prefix}_error_vs_pt.png")

    return {"errors": errors, "calibration": calib}
