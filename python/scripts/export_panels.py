"""Export EOD panels to the raw binary format read by the C++ engine.

Cleaning is delegated to ``gp_factor_miner.data.load_eod_panel_dataset`` so the
exported values match what the Python GP evaluates (features are only aligned
and cast to float32; the target is masked to tradable, non-ST rows).

Output layout (see docs/data_format.md)::

    <out>/meta.json             T, N, dates, assets, fields, dtype, layout
    <out>/fields/<field>.f32    float32, date-major [T][N], little-endian
    <out>/target.f32            float32, date-major [T][N]

Example::

    PYTHONPATH=src:/home/research_pyx/GP/.deps python3 scripts/export_panels.py \
        --data-file /hy-tmp/autodl-tmp/EOD_Columns/eod \
        --label-file /hy-tmp/data/factor/label_10.pkl \
        --out /hy-tmp/data_pyx/GP_distributed/sample_120x300 \
        --period evolution --max-dates 120 --max-assets 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gp_factor_miner.data import load_eod_panel_dataset, split_eod_panels


def _write_panel(path: Path, frame) -> None:
    values = np.ascontiguousarray(frame.to_numpy(dtype="<f4"))
    values.tofile(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--period", choices=("evolution", "screening"), default="evolution")
    parser.add_argument("--max-dates", type=int, default=None, help="keep the first k dates")
    parser.add_argument("--max-assets", type=int, default=None, help="keep the first k assets")
    parser.add_argument("--fields", nargs="*", default=None, help="subset of fields (default: all)")
    args = parser.parse_args(argv)

    dataset = load_eod_panel_dataset(args.data_file, args.label_file)
    evolution, evolution_target, screening, screening_target = split_eod_panels(dataset)
    features, target = (
        (evolution, evolution_target) if args.period == "evolution" else (screening, screening_target)
    )

    fields = sorted(args.fields) if args.fields else sorted(features)
    unknown = [name for name in fields if name not in features]
    if unknown:
        raise SystemExit(f"unknown fields: {unknown}")

    dates = list(target.index)[: args.max_dates]
    assets = list(target.columns)[: args.max_assets]
    target = target.loc[dates, assets]

    out = Path(args.out)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    for name in fields:
        _write_panel(out / "fields" / f"{name}.f32", features[name].loc[dates, assets])
    _write_panel(out / "target.f32", target)

    meta = {
        "format_version": 1,
        "period": args.period,
        "T": len(dates),
        "N": len(assets),
        "dtype": "float32",
        "layout": "date_major",
        "dates": [str(value) for value in dates],
        "assets": [str(value) for value in assets],
        "fields": fields,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(fields)} fields, T={len(dates)}, N={len(assets)} to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
