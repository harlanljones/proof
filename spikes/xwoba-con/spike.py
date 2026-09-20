"""SPIKE (throwaway): does YTD xwOBAcon add ROS-FIP signal beyond proof?

Question: proof already blends YTD SO/uBBHBP/HR rates toward a prior. If
contact quality carried extra signal, residual (actual - proof) would
correlate with YTD xwOBAcon after shrinking for BBE noise.

Method, leakage-safe:
  * YTD xwOBAcon per pitcher per checkpoint from pinned pitch-level
    snapshots (game_date <= checkpoint), mean estimated_woba_using_speedangle
    over batted balls. League mean regressed by k BBE (grid 100..800).
  * Tune shrink k and correction weight b on 2023-24 ONLY; evaluate on 2025.
  * pred' = pred_fip_proof + b * (xw_shrunk - lg_xw)
"""

import numpy as np
import pandas as pd

from backtest.checkpoints import grid
from ingest import snapshots

SEASON_BOUNDS = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-18", "2025-09-28"),
}

IDENTITY_POOLS = [2023, 2025]
MIN_BBE = 25  # YTD BBE floor: below this, xwOBAcon is noise
MIN_BF_ROS = 80


def season_rows(season: int) -> pd.DataFrame:
    """Raw pitch-level snapshot rows (pitcher = MLB id, game_date string)."""
    return snapshots.load(f"pitchlevel_{season}")


def con_table(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    w = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
    bb = w[w["estimated_woba_using_speedangle"].notna()]
    g = bb.groupby("pitcher")["estimated_woba_using_speedangle"].agg(["mean", "size"])
    g.columns = ["xw_con", "bbe"]
    g = g[g["bbe"] >= MIN_BBE]
    return g.reset_index().rename(columns={"pitcher": "mlbID"})


def shrink(t: pd.DataFrame, lg: float, k: float) -> pd.Series:
    return (t["bbe"] * t["xw_con"] + k * lg) / (t["bbe"] + k)


def evaluate(seasons, k, lg_by_year, preds):
    rows = []
    for season in seasons:
        full = season_rows(season)
        lg = lg_by_year[season]
        for cp in grid(season):
            ytd = con_table(full, cp["ytd_start"], cp["ytd_end"])
            cell = preds[
                (preds["season"] == season) & (preds["checkpoint"] == cp["checkpoint"])
            ]
            m = cell.merge(ytd, on="mlbID", how="inner")
            xw_s = shrink(m, lg, k)
            corr = np.corrcoef(xw_s, m["fip_actual_ros"] - m["pred_fip_proof"])[0, 1]
            rmse0 = np.sqrt(
                np.average((m["pred_fip_proof"] - m["fip_actual_ros"]) ** 2)
            )
            rows.append(
                {
                    "season": season,
                    "checkpoint": cp["checkpoint"],
                    "n": len(m),
                    "corr_xw_resid": corr,
                    "rmse_proof": rmse0,
                    "xw_con": xw_s.values,
                    "pred0": m["pred_fip_proof"].values,
                    "actual": m["fip_actual_ros"].values,
                    "bf": m["BF_ros"].values,
                }
            )
    return rows


def main():
    preds = pd.read_parquet("writeup/results/pitchers_fip_predictions.parquet")
    # League xwOBAcon by season (observational, from YTD window at each
    # checkpoint in production; whole-season here is a spike shortcut that
    # slightly leaks league level, not player signal).
    lg_by_year = {}
    for s in (2023, 2024, 2025):
        full = season_rows(s)
        t = con_table(full, f"{s}-04-01", f"{s}-10-01")
        lg_by_year[s] = float(np.average(t["xw_con"], weights=t["bbe"]))
        print(f"lg xwOBAcon {s}: {lg_by_year[s]:.4f} (n={len(t)} pitchers)")

    # --- training: 2023-24, grid over k and b
    tr_rows = evaluate([2023, 2024], 400.0, lg_by_year, preds)  # corr only
    c = pd.DataFrame(
        [
            {k: r[k] for k in ("season", "checkpoint", "corr_xw_resid", "n")}
            for r in tr_rows
        ]
    )
    print("\nresidual corr (k=400, descriptive per cell):")
    print(c.to_string(index=False))

    best = None
    for k in (100, 200, 400, 800):
        rows = evaluate([2023, 2024], k, lg_by_year, preds)
        # joint fit: concatenate standardized cells (per-cell centered)
        X = np.concatenate(
            [
                (r["xw_con"] - np.average(r["xw_con"], weights=r["bf"]))
                / np.sqrt(r["n"] + k)
                for r in rows
            ]
        )
        Y = np.concatenate([(r["actual"] - r["pred0"]) for r in rows])
        b = float(np.dot(X, Y) / np.dot(X, X))
        rmse = float(np.sqrt(np.mean((Y - b * X) ** 2)))
        rmse0 = float(np.sqrt(np.mean(Y**2)))
        print(f"k={k:4d}: b={b:+.4f}  train RMSE {rmse0:.5f} -> {rmse:.5f}")
        if best is None or rmse < best[2]:
            best = (k, b, rmse)
    k_star, b_star, _ = best
    print(f"\nselected on 2023-24: k={k_star}, b={b_star:+.4f}")

    # --- holdout: 2025
    ho_rows = evaluate([2025], k_star, lg_by_year, preds)
    X = np.concatenate(
        [
            (r["xw_con"] - np.average(r["xw_con"], weights=r["bf"]))
            / np.sqrt(r["n"] + k_star)
            for r in ho_rows
        ]
    )
    Y = np.concatenate([(r["actual"] - r["pred0"]) for r in ho_rows])
    rmse0 = float(np.sqrt(np.mean(Y**2)))
    rmse1 = float(np.sqrt(np.mean((Y - b_star * X) ** 2)))
    print(f"holdout 2025 RMSE: proof {rmse0:.5f} -> +xwOBAcon {rmse1:.5f}")
    print(f"holdout improvement: {100 * (rmse0 - rmse1) / rmse0:+.3f}%")
    print(f"holdout residual corr with shrunk xw: {np.corrcoef(X, Y)[0, 1]:+.4f}")


if __name__ == "__main__":
    main()
