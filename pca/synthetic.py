#!/usr/bin/env python3
"""Generate documented controlled-spectrum inputs, never presented as market data."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=293)
    parser.add_argument("--cols", type=int, default=5876)
    parser.add_argument("--factors", type=int, default=24)
    parser.add_argument("--noise", type=float, default=1e-8)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to((Path(__file__).resolve().parent / "runs").resolve()):
        parser.error("output must be new and inside runs/")
    m, n = args.rows, args.cols
    if min(m, n) < 6 or args.factors < 1 or args.noise < 0:
        parser.error("rows/cols >=6, factors positive, noise nonnegative")
    args.out.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    initial = np.random.default_rng(0).normal(size=n)
    initial.astype("<f8").tofile(args.out / "initial.f8")
    records, descriptions = [], []
    chunk = args.out / "R.f32"
    # DCT columns (nonconstant modes) are analytically orthonormal; no QR/SVD
    # or numerical-library matrix products are needed for input construction.
    with chunk.open("wb") as output:
        for b in range(args.factors):
            gap = (.5, .9, .99)[b % 3]
            singular = (1., gap, .2, .1)
            frequencies_u = rng.choice(np.arange(1, m), size=4, replace=False)
            frequencies_v = rng.choice(np.arange(1, n), size=4, replace=False)
            R = np.zeros((m, n))
            for sigma, fu, fv in zip(singular, frequencies_u, frequencies_v):
                u = np.sqrt(2/m) * np.cos(np.pi * (np.arange(m) + .5) * fu/m)
                v = np.sqrt(2/n) * np.cos(np.pi * (np.arange(n) + .5) * fv/n)
                R += sigma * u[:, None] * v[None, :]
            R += args.noise * rng.normal(size=(m, n))
            R.astype("<f4").tofile(output)
            records.append(dict(index=b, score_line=b, key=f"synthetic_{b:06d}",
                                quality=.03 + .01 * (b % 5), candidate=True,
                                path=str(chunk.resolve()), slot=b))
            descriptions.append(dict(factor=b, sigma2_over_sigma1_before_noise=gap,
                                     left_DCT_frequencies=frequencies_u.tolist(),
                                     right_DCT_frequencies=frequencies_v.tolist()))
    manifest = dict(rows=m, cols=n, initial=str((args.out / "initial.f8").resolve()),
                    records=records, synthetic=dict(seed=2026, noise_std=args.noise,
                    construction="sum of four analytically orthonormal DCT outer products + Gaussian noise",
                    note="controlled spectral matrices, not percentile-ranked market factors",
                    factors=descriptions))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {args.factors} controlled-spectrum matrices of shape {m} x {n}")


if __name__ == "__main__":
    main()
