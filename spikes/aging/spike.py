"""SPIKE (throwaway): is Marcel's crude aging scale right for pitchers?

models.pitchers ages arms by f (Marcel) and scales K by f, uBB/HR by 1/f,
calling itself "crude — empirical aging curves are phase 2". Before building
an empirical curve, test whether the aging dial matters at all: replace
f**1 with f**m and walk-forward. If the curve is flat around m=1 (or m=0
wins), empirical aging is not worth building.

Leakage: m selected on 2023-24 only; 2025 evaluated once.
"""

import numpy as np
import pandas as pd

import models.pitchers as mp
from backtest.run import K_MULT
from backtest.walkforward_pitchers import run_backtest_pitchers_fip

ORIGINAL_AGING = mp.aging_factor


def make_power_aging(m: float):
    base = ORIGINAL_AGING

    def f(age):
        return base(age) ** m

    f.__name__ = f"aging_f^{m}"
    return f


def wrmse_proof(results: pd.DataFrame) -> float:
    err = (results["pred_fip_proof"] - results["fip_actual_ros"]) ** 2
    return float(np.sqrt(np.average(err, weights=results["BF_ros"])))


def run_with(m: float, seasons: list[int]) -> tuple[float, int]:
    mp.aging_factor = make_power_aging(m)
    res = run_backtest_pitchers_fip(seasons, k_mult=K_MULT)
    return wrmse_proof(res), len(res)


def main():
    train = {}
    for m in (0.0, 0.5, 1.0, 1.5):
        v, n = run_with(m, [2023, 2024])
        train[m] = v
        print(f"m={m:.1f}  train 2023-24 wRMSE proof = {v:.5f}  (n={n})")
    m_star = min(train, key=train.get)
    print(f"\nselected on 2023-24: m={m_star}")

    v, n = run_with(m_star, [2025])
    print(f"holdout 2025 wRMSE proof @ m* = {v:.5f}  (n={n})")
    v1, _ = run_with(1.0, [2025])
    print(f"holdout 2025 wRMSE proof @ m=1 (Marcel) = {v1:.5f}")


if __name__ == "__main__":
    main()
