"""Synthetic panels in the export format (smoke runs and tests)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .export import write_panels

FIELDS = ("close", "open", "volume", "amount", "turnover", "pe")


def make_frames(T, N, seed, missing=0.03, start="2016-01-04"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=T).strftime("%Y%m%d")
    assets = [f"{600000 + i:06d}" for i in range(N)]
    base = rng.normal(size=(T, N)).cumsum(axis=0) * 0.1
    frames = {}
    for k, name in enumerate(FIELDS):
        values = base * (1 + 0.3 * k) + rng.normal(size=(T, N)) * (0.5 + 0.1 * k) + 0.05 * k * np.arange(N)
        if name in ("volume", "amount"):
            values = np.exp(values * 0.2)
        values[rng.random((T, N)) < missing] = np.nan
        frames[name] = pd.DataFrame(values.astype(np.float32), index=dates, columns=assets)
    future = base[np.r_[1:T, T - 1]] - base
    target = 0.6 * future + 0.2 * frames["close"].to_numpy(np.float64) * 0.01 + rng.normal(size=(T, N)) * 0.05
    target[rng.random((T, N)) < missing] = np.nan
    target = pd.DataFrame(target.astype(np.float32), index=dates, columns=assets)
    return frames, target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--dates", type=int, default=300)
    p.add_argument("--stocks", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--period", choices=("evolution", "screening"), default="evolution")
    a = p.parse_args(argv)
    start = "2016-01-04" if a.period == "evolution" else "2022-01-04"
    frames, target = make_frames(a.dates, a.stocks, a.seed, start=start)
    write_panels(a.out, frames, target, period=a.period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
