"""`python -m backtest.run` — the reproducible backtest (`make backtest`).

Modes:
  default          hitter walk-forward 2023-2025 at the tuned shrinkage
                   multiplier, metrics -> writeup/results/hitters_*
  --pitchers       pitcher walk-forward 2023-2025 -> writeup/results/pitchers_*
  --tune           choose the hitter k_mult on 2023-2024 only, then evaluate
                   on the 2025 holdout the tuning loop never saw
  --seasons ...    custom season list (k_mult fixed at --k-mult)
"""

from __future__ import annotations

import argparse

import pandas as pd

from backtest.metrics import RESULTS_DIR, write_all
from backtest.tune import (
    HOLDOUT_SEASONS,
    TRAIN_SEASONS,
    tune,
    tune_pitchers,
    weighted_rmse,
)
from backtest.walkforward import SYSTEMS, run_backtest
from backtest.walkforward_pitchers import (
    SYSTEMS_P,
    run_backtest_pitchers,
    run_backtest_pitchers_fip,
)

DEFAULT_SEASONS = [2023, 2024, 2025]
# Chosen by `python -m backtest.run --tune` (grid on 2023-24, validated on the
# 2025 holdout it never saw; curve is flat past 4.0 -> interior optimum).
# See writeup/results/tuning.csv. 1.0 = borrowed Carleton constants.
K_MULT = 4.0
# Pitcher equivalent, same protocol (writeup/results/tuning_pitchers.csv).
K_MULT_P = 2.0


def _print(summary: dict, target: str) -> None:
    with pd.option_context(
        "display.float_format", "{:.4f}".format, "display.width", 120
    ):
        print(f"\n=== RMSE / MAE on rest-of-season {target} ===")
        print(summary["errors"].to_string(index=False))
        print("\n=== Calibration (nominal vs empirical interval coverage) ===")
        print(summary["calibration"].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--k-mult", type=float, default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-pitchers", action="store_true")
    ap.add_argument("--pitchers", action="store_true", help="FIP walk-forward (Savant)")
    ap.add_argument(
        "--pitchers-era",
        action="store_true",
        help="ERA walk-forward (BR actuals; only 2023 inputs are pinned)",
    )
    args = ap.parse_args()

    if args.tune:
        print(f"Tuning k_mult on {TRAIN_SEASONS} (holdout {HOLDOUT_SEASONS} untouched)")
        table, best = tune()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS_DIR / "tuning.csv", index=False)
        print(f"\n{table.to_string(index=False)}\nselected k_mult = {best}")

        print(f"\nHeld-out evaluation on {HOLDOUT_SEASONS}:")
        for km in sorted({1.0, best}):
            res = run_backtest(HOLDOUT_SEASONS, k_mult=km)
            print(
                f"  k_mult={km}: holdout wRMSE proof={weighted_rmse(res):.5f} "
                f"prior={weighted_rmse(res, 'prior'):.5f} "
                f"naive={weighted_rmse(res, 'naive'):.5f}"
            )
        return

    if args.tune_pitchers:
        print(
            f"Tuning pitcher k_mult on {TRAIN_SEASONS} "
            f"(holdout {HOLDOUT_SEASONS} untouched)"
        )
        table, best = tune_pitchers()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS_DIR / "tuning_pitchers.csv", index=False)
        print(f"\n{table.to_string(index=False)}\nselected k_mult = {best}")

        print(f"\nHeld-out evaluation on {HOLDOUT_SEASONS}:")
        for km in sorted({1.0, best}):
            res = run_backtest_pitchers_fip(HOLDOUT_SEASONS, k_mult=km)
            print(
                f"  k_mult={km}: holdout wRMSE proof={weighted_rmse(res, 'proof', 'fip', 'BF_ros'):.5f} "
                f"prior={weighted_rmse(res, 'prior', 'fip', 'BF_ros'):.5f} "
                f"eb_league={weighted_rmse(res, 'eb_league', 'fip', 'BF_ros'):.5f} "
                f"naive={weighted_rmse(res, 'naive', 'fip', 'BF_ros'):.5f}"
            )
        return

    if args.pitchers:
        k_mult = args.k_mult if args.k_mult is not None else K_MULT_P
        print(
            f"PROOF pitcher walk-forward (FIP target), seasons: {args.seasons}, "
            f"k_mult={k_mult}"
        )
        results = run_backtest_pitchers_fip(args.seasons, k_mult=k_mult)
        print(
            f"{len(results)} pitcher-checkpoint rows "
            f"({results['mlbID'].nunique()} unique pitchers)"
        )
        _print(write_all(results, "fip", SYSTEMS_P, "BF_ros", "pitchers_fip"), "fip")
        print("\nWrote writeup/results/pitchers_fip_*")
        return

    if args.pitchers_era:
        k_mult = args.k_mult if args.k_mult is not None else K_MULT_P
        print(f"PROOF pitcher walk-forward (ERA target, BR), seasons: {args.seasons}")
        results = run_backtest_pitchers(args.seasons, k_mult=k_mult)
        print(
            f"{len(results)} pitcher-checkpoint rows "
            f"({results['mlbID'].nunique()} unique pitchers)"
        )
        _print(write_all(results, "era", SYSTEMS_P, "BF_ros", "pitchers_era"), "era")
        print(
            "\nNOTE: ERA intervals = FIP interval + defense/sequencing drift"
            " variance (models.pitchers.SIGMA_ERA_EXTRA2, estimated on"
            " 2023-24 training residuals only)."
        )
        print("\nWrote writeup/results/pitchers_era_*")
        return

    k_mult = args.k_mult if args.k_mult is not None else K_MULT
    print(f"PROOF hitter walk-forward, seasons: {args.seasons}, k_mult={k_mult}")
    results = run_backtest(args.seasons, k_mult=k_mult)
    print(
        f"{len(results)} player-checkpoint rows "
        f"({results['mlbID'].nunique()} unique players)"
    )
    _print(write_all(results, "woba", SYSTEMS, "PA_ros", "hitters"), "woba")
    print("\nWrote writeup/results/hitters_*")


if __name__ == "__main__":
    main()
