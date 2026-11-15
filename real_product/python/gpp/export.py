"""Export evolution / screening panels for libgpp_native.

Same binary layout and cleaning as python/scripts/export_panels.py
(docs/engine/data_format.md). One addition: ``--warmup-days K`` for the
screening period prepends the last K evolution dates, and meta.json records
``score_begin = K`` so windows see real history while RankIC and the
correlation vectors still cover 2022 only. K = 0 (default) reproduces the
course semantics exactly (windows restart on the first 2022 day).

    python -m gpp.export --data-file EOD --label-file LABEL --out DIR \
        --period screening --warmup-days 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gp_factor_miner.data import load_eod_panel_dataset, split_eod_panels


def write_panels(out, features, target, *, period, score_begin=0, fields=None):
    out = Path(out)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    fields = sorted(fields or features)
    dates, assets = list(target.index), list(target.columns)
    for name in fields:
        values = features[name].reindex(index=dates, columns=assets)
        np.ascontiguousarray(values.to_numpy(dtype="<f4")).tofile(out / "fields" / f"{name}.f32")
    np.ascontiguousarray(target.to_numpy(dtype="<f4")).tofile(out / "target.f32")
    meta = {
        "format_version": 1,
        "period": period,
        "T": len(dates),
        "N": len(assets),
        "dtype": "float32",
        "layout": "date_major",
        "dates": [str(x) for x in dates],
        "assets": [str(x) for x in assets],
        "fields": fields,
        "score_begin": int(score_begin),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--period", choices=("evolution", "screening"), default="evolution")
    parser.add_argument("--warmup-days", type=int, default=0)
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--max-assets", type=int)
    args = parser.parse_args(argv)

    dataset = load_eod_panel_dataset(args.data_file, args.label_file)
    evolution, evolution_target, screening, screening_target = split_eod_panels(dataset)
    if args.period == "evolution":
        features, target, begin = evolution, evolution_target, 0
    else:
        k = max(0, args.warmup_days)
        history = list(evolution_target.index)[-k:] if k else []
        if k:
            features = {name: pd.concat([evolution[name].loc[history], screening[name]]) for name in screening}
            # Warm-up rows carry no label: they are never scored.
            target = pd.concat([evolution_target.loc[history] * np.nan, screening_target])
        else:
            features, target = screening, screening_target
        begin = len(history)
    dates = list(target.index)
    if args.max_dates:
        dates = dates[: begin + args.max_dates]
    assets = list(target.columns)[: args.max_assets]
    meta = write_panels(args.out, features, target.loc[dates, assets], period=args.period, score_begin=begin)
    print(f"wrote {len(meta['fields'])} fields T={meta['T']} N={meta['N']} score_begin={begin} to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
